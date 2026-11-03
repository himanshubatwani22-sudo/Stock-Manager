import hashlib
import math
import re
import secrets
import sqlite3
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from features import make_features

st.set_page_config(page_title="Warehouse Demand Predictor", page_icon="📦", layout="centered")

SAFETY_PCT = 0.20   # safety stock = 20% of demand during supplier lead time (assumption)
COVER_DAYS = 14     # extra days of sales to cover when suggesting an order

# ---------------------------- database (users + products)
DB_PATH = "users.db"


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "email TEXT PRIMARY KEY, salt TEXT NOT NULL, pw_hash TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS products ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, sku TEXT NOT NULL, name TEXT NOT NULL, "
        "unit_price REAL NOT NULL, stock INTEGER NOT NULL, lead_time INTEGER NOT NULL, "
        "s1 REAL NOT NULL, s2 REAL NOT NULL, s3 REAL NOT NULL, s4 REAL NOT NULL, UNIQUE(owner, sku))"
    )
    return conn


def hash_pw(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def create_user(email, password):
    salt = secrets.token_hex(16)
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users (email, salt, pw_hash) VALUES (?, ?, ?)",
                (email.lower(), salt, hash_pw(password, salt)),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def check_user(email, password):
    with db() as conn:
        row = conn.execute(
            "SELECT salt, pw_hash FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
    if not row:
        return False
    return secrets.compare_digest(hash_pw(password, row[0]), row[1])


def add_product(owner, sku, name, price, stock, lead_time, sales):
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO products (owner, sku, name, unit_price, stock, lead_time, s1, s2, s3, s4) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (owner, sku, name, price, stock, lead_time, *sales),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def update_product(owner, pid, name, price, stock, lead_time, sales):
    with db() as conn:
        conn.execute(
            "UPDATE products SET name=?, unit_price=?, stock=?, lead_time=?, s1=?, s2=?, s3=?, s4=? "
            "WHERE id=? AND owner=?",
            (name, price, stock, lead_time, *sales, pid, owner),
        )


def delete_product(owner, pid):
    with db() as conn:
        conn.execute("DELETE FROM products WHERE id=? AND owner=?", (pid, owner))


def get_products(owner):
    with db() as conn:
        return conn.execute(
            "SELECT id, sku, name, unit_price, stock, lead_time, s1, s2, s3, s4 "
            "FROM products WHERE owner = ? ORDER BY name", (owner,)
        ).fetchall()


# ---------------------------- model + calculations
@st.cache_resource
def load_model():
    return joblib.load("demand_best_model.pkl")


def predict_demand(lags, prices, festival):
    """lags: list of [s1,s2,s3,s4]; returns predicted units/day for each row (same logic as before)."""
    x, avg = make_features(lags, prices, festival)
    ratio = np.clip(load_model().predict(x), 0, None)
    return np.where(avg == 0, 0.1, np.maximum(ratio * avg, 0.1))


def calc_metrics(demand, stock, lead_time, price):
    """All reorder numbers from the predicted daily demand."""
    days_left = stock / demand
    lead_demand = demand * lead_time
    safety_stock = lead_demand * SAFETY_PCT
    reorder_point = lead_demand + safety_stock
    if days_left <= lead_time:
        status = "critical"
    elif stock <= reorder_point:
        status = "soon"
    else:
        status = "ok"
    order_qty = 0
    if status != "ok":
        order_qty = max(0, math.ceil(reorder_point + demand * COVER_DAYS - stock))
    return {
        "days_left": days_left,
        "stockout_date": date.today() + timedelta(days=int(days_left)),
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "status": status,
        "order_qty": order_qty,
        "weekly_units": demand * 7,
        "weekly_revenue": demand * 7 * price,
    }


STATUS_LABEL = {"critical": "🔴 Reorder now", "soon": "🟡 Reorder soon", "ok": "🟢 OK"}


def validate_product(sku, name, sales, existing_skus=(), editing_sku=None):
    errors = []
    if not sku.strip():
        errors.append("Product code (SKU) is required.")
    if not name.strip():
        errors.append("Product name is required.")
    if (sku.strip() and sku.strip().lower() in [s.lower() for s in existing_skus]
            and sku.strip() != editing_sku):
        errors.append("A product with this code already exists.")
    if all(s == 0 for s in sales):
        errors.append("All past sales are zero. Enter real sales so the prediction is meaningful.")
    return errors


# ---------------------------- session
if "user" not in st.session_state:
    st.session_state.user = None


# ---------------------------- login and sign-up page
def auth_page():
    st.title("Warehouse Demand Predictor")
    st.caption("Machine learning app that forecasts next week's sales and warns when to reorder.")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
        if submit:
            if check_user(email.strip(), password):
                st.session_state.user = email.strip().lower()
                st.rerun()
            else:
                st.error("Incorrect email or password.")

    with tab_signup:
        with st.form("signup"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_pw")
            confirm = st.text_input("Confirm password", type="password")
            create = st.form_submit_button("Create account", use_container_width=True)
        if create:
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", new_email.strip()):
                st.error("Enter a valid email address.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_password != confirm:
                st.error("Passwords don't match.")
            elif create_user(new_email.strip(), new_password):
                st.success("Account created. Open the Login tab to sign in.")
            else:
                st.error("An account with this email already exists.")


# ---------------------------- products page (add / edit / delete + status)
WEEK_LABELS = ["Last week", "2 weeks ago", "3 weeks ago", "4 weeks ago"]


def products_page():
    user = st.session_state.user
    st.title("My Products")
    st.caption("Add the products in your warehouse. You can load them on the prediction page.")

    rows = get_products(user)
    tab_add, tab_edit = st.tabs(["➕ Add product", "✏️ Edit / Delete"])

    # ---- add
    with tab_add:
        with st.form("add_product", clear_on_submit=True):
            c1, c2 = st.columns(2)
            sku = c1.text_input("Product code (SKU)")
            name = c2.text_input("Product name")
            c3, c4, c5 = st.columns(3)
            price = c3.number_input("Unit price", 0.1, 1000.0, 2.0)
            stock = c4.number_input("Current stock (units)", 0, 1000000, 100)
            lead_time = c5.number_input("Supplier lead time (days)", 1, 60, 7)
            st.write("Average units sold per day, in each past week")
            w = st.columns(4)
            defaults = [10.0, 12.0, 8.0, 11.0]
            sales = [w[i].number_input(WEEK_LABELS[i], 0.0, 100000.0, defaults[i]) for i in range(4)]
            add = st.form_submit_button("Add product", use_container_width=True)

        if add:
            errors = validate_product(sku, name, sales, [r[1] for r in rows])
            if errors:
                for e in errors:
                    st.error(e)
            elif add_product(user, sku.strip(), name.strip(), price, stock, lead_time, sales):
                st.success(f"Added {name.strip()}.")
                st.rerun()
            else:
                st.error("A product with this code already exists.")

    # ---- edit / delete
    with tab_edit:
        if not rows:
            st.info("Nothing to edit yet.")
        else:
            labels = {r[0]: f"{r[1]} - {r[2]}" for r in rows}
            pid = st.selectbox("Select a product", list(labels), format_func=labels.get)
            r = next(x for x in rows if x[0] == pid)
            with st.form(f"edit_{pid}"):
                st.text_input("Product code (SKU)", r[1], disabled=True)
                new_name = st.text_input("Product name", r[2])
                c3, c4, c5 = st.columns(3)
                e_price = c3.number_input("Unit price", 0.1, 1000.0, float(r[3]))
                e_stock = c4.number_input("Current stock (units)", 0, 1000000, int(r[4]))
                e_lead = c5.number_input("Supplier lead time (days)", 1, 60, int(r[5]))
                st.write("Average units sold per day, in each past week")
                w = st.columns(4)
                e_sales = [w[i].number_input(WEEK_LABELS[i], 0.0, 100000.0, float(r[6 + i])) for i in range(4)]
                save = st.form_submit_button("Save changes", use_container_width=True)
            if save:
                errors = validate_product(r[1], new_name, e_sales, [], editing_sku=r[1])
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    update_product(user, pid, new_name.strip(), e_price, e_stock, e_lead, e_sales)
                    st.success("Saved.")
                    st.rerun()

            confirm = st.checkbox(f"I want to delete {labels[pid]}")
            if st.button("Delete product", disabled=not confirm):
                delete_product(user, pid)
                st.rerun()

    # ---- saved products table with live status
    st.divider()
    if rows:
        st.subheader(f"Saved products ({len(rows)})")
        try:
            lags = [[r[6], r[7], r[8], r[9]] for r in rows]
            demand = predict_demand(lags, [r[3] for r in rows], [0] * len(rows))
            status = [STATUS_LABEL[calc_metrics(d, r[4], r[5], r[3])["status"]] for d, r in zip(demand, rows)]
        except FileNotFoundError:
            demand, status = [None] * len(rows), ["n/a"] * len(rows)
        st.dataframe(
            [{"Code": r[1], "Product": r[2], "Price": r[3], "Stock": r[4],
              "Lead time (days)": r[5],
              "Predicted/day": None if d is None else round(float(d), 1), "Status": s}
             for r, d, s in zip(rows, demand, status)],
            use_container_width=True, hide_index=True,
        )
        st.caption("Status uses the model's prediction for a normal (non-festival) week.")
    else:
        st.info("No products yet.")


# ---------------------------- prediction page
def predict_page():
    st.title("Demand and Reorder Prediction")
    st.caption("Enter the last 4 weeks of sales. The model predicts next week's sales, then checks if you must reorder.")

    # optional: fill the form from a saved product
    saved = get_products(st.session_state.user)
    pick = 0
    if saved:
        names = {r[0]: f"{r[1]} - {r[2]}" for r in saved}
        pick = st.selectbox("Load a saved product (optional)", [0] + list(names),
                            format_func=lambda i: "Enter manually" if i == 0 else names[i])
    row = next((r for r in saved if r[0] == pick), None)
    d = dict(price=2.0, stock=100, lead=7, s1=10.0, s2=12.0, s3=8.0, s4=11.0)
    if row:
        d = dict(price=float(row[3]), stock=int(row[4]), lead=int(row[5]),
                 s1=float(row[6]), s2=float(row[7]), s3=float(row[8]), s4=float(row[9]))

    with st.form("predict"):
        st.subheader("Average units sold per day")
        c1, c2 = st.columns(2)
        s1 = c1.number_input("Last week", 0.0, 100000.0, d["s1"], key=f"s1_{pick}")
        s2 = c2.number_input("2 weeks ago", 0.0, 100000.0, d["s2"], key=f"s2_{pick}")
        c3, c4 = st.columns(2)
        s3 = c3.number_input("3 weeks ago", 0.0, 100000.0, d["s3"], key=f"s3_{pick}")
        s4 = c4.number_input("4 weeks ago", 0.0, 100000.0, d["s4"], key=f"s4_{pick}")

        st.subheader("Product and warehouse")
        price = st.number_input("Unit price", 0.1, 1000.0, d["price"], key=f"price_{pick}")
        festival = st.checkbox("Next week is festival season")
        c5, c6 = st.columns(2)
        stock = c5.number_input("Current stock (units)", 0, 1000000, d["stock"], key=f"stock_{pick}")
        lead_time = c6.number_input("Supplier lead time (days)", 1, 60, d["lead"], key=f"lead_{pick}")

        submit = st.form_submit_button("Predict", use_container_width=True)

    if submit:
        if all(v == 0 for v in (s1, s2, s3, s4)):
            st.error("All past sales are zero. Enter real sales data to get a prediction.")
            return
        try:
            demand = float(predict_demand([[s1, s2, s3, s4]], [price], [1 if festival else 0])[0])
        except FileNotFoundError:
            st.error("demand_best_model.pkl not found. Run train_models.py first.")
            return

        m = calc_metrics(demand, stock, lead_time, price)

        st.divider()
        if m["status"] == "critical":
            st.error(f"REORDER NOW: stock runs out before the supplier can deliver. Suggested order: {m['order_qty']} units.")
        elif m["status"] == "soon":
            st.warning(f"REORDER SOON: stock is below the reorder point. Suggested order: {m['order_qty']} units.")
        else:
            st.success("Stock level is OK. No reorder needed yet.")

        a1, a2 = st.columns(2)
        a1.metric("Predicted sales next week", f"{demand:.1f} units/day")
        a2.metric("Predicted total (7 days)", f"{m['weekly_units']:.0f} units")
        b1, b2 = st.columns(2)
        b1.metric("Stock will last about", f"{m['days_left']:.1f} days")
        b2.metric("Expected stock-out date", m["stockout_date"].strftime("%d %b %Y"))
        c1, c2 = st.columns(2)
        c1.metric("Reorder point", f"{m['reorder_point']:.0f} units")
        c2.metric("Safety stock", f"{m['safety_stock']:.0f} units")
        d1, d2 = st.columns(2)
        d1.metric("Suggested order quantity", f"{m['order_qty']} units")
        d2.metric("Expected revenue (7 days)", f"{m['weekly_revenue']:,.2f}")

        with st.expander("How these numbers are calculated"):
            st.markdown(
                "- **Stock will last** = current stock / predicted daily sales\n"
                f"- **Safety stock** = {int(SAFETY_PCT * 100)}% of the demand during the supplier lead time (assumption)\n"
                "- **Reorder point** = predicted daily sales x lead time + safety stock\n"
                "- **Reorder now** = stock runs out within the lead time; **Reorder soon** = stock is at or below the reorder point\n"
                f"- **Suggested order** = reorder point + {COVER_DAYS} days of sales - current stock"
            )


# ---------------------------- routing
if st.session_state.user:
    with st.sidebar:
        st.header("Account")
        st.write(f"Signed in as {st.session_state.user}")
        page = st.radio("Page", ["Demand Prediction", "My Products"])
        if st.button("Log out", use_container_width=True):
            st.session_state.user = None
            st.rerun()
    if page == "My Products":
        products_page()
    else:
        predict_page()
else:
    auth_page()
