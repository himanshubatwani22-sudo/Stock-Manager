import hashlib
import re
import secrets
import sqlite3
import joblib
import numpy as np
import streamlit as st
from features import make_features

st.set_page_config(page_title="Warehouse Demand Predictor", page_icon="📦", layout="centered")

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


def get_products(owner):
    with db() as conn:
        return conn.execute(
            "SELECT id, sku, name, unit_price, stock, lead_time, s1, s2, s3, s4 "
            "FROM products WHERE owner = ? ORDER BY name", (owner,)
        ).fetchall()


# ---------------------------- model
@st.cache_resource
def load_model():
    return joblib.load("demand_best_model.pkl")


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


# ---------------------------- products page (multiple product management)
def products_page():
    st.title("My Products")
    st.caption("Add the products in your warehouse. You can load them on the prediction page.")

    with st.form("add_product", clear_on_submit=True):
        c1, c2 = st.columns(2)
        sku = c1.text_input("Product code (SKU)")
        name = c2.text_input("Product name")
        c3, c4, c5 = st.columns(3)
        price = c3.number_input("Unit price", 0.1, 1000.0, 2.0)
        stock = c4.number_input("Current stock (units)", 0, 1000000, 100)
        lead_time = c5.number_input("Supplier lead time (days)", 1, 60, 7)
        st.write("Past sales (units per day)")
        w = st.columns(4)
        s1 = w[0].number_input("Last week", 0.0, 100000.0, 10.0)
        s2 = w[1].number_input("2 weeks ago", 0.0, 100000.0, 12.0)
        s3 = w[2].number_input("3 weeks ago", 0.0, 100000.0, 8.0)
        s4 = w[3].number_input("4 weeks ago", 0.0, 100000.0, 11.0)
        add = st.form_submit_button("Add product", use_container_width=True)

    if add:
        if not sku.strip() or not name.strip():
            st.error("Product code and name are required.")
        elif add_product(st.session_state.user, sku.strip(), name.strip(), price, stock, lead_time, [s1, s2, s3, s4]):
            st.success(f"Added {name.strip()}.")
        else:
            st.error("A product with this code already exists.")

    rows = get_products(st.session_state.user)
    st.divider()
    if rows:
        st.subheader(f"Saved products ({len(rows)})")
        st.dataframe(
            [{"Code": r[1], "Product": r[2], "Price": r[3], "Stock": r[4], "Lead time (days)": r[5]} for r in rows],
            use_container_width=True, hide_index=True,
        )
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
        st.subheader("Past sales (units per day)")
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
        try:
            model = load_model()
        except FileNotFoundError:
            st.error("demand_best_model.pkl not found. Run train_models.py first.")
            return

        x, avg = make_features([[s1, s2, s3, s4]], [price], [1 if festival else 0])
        if avg[0] == 0:
            demand = 0.1
        else:
            demand = max(model.predict(x)[0] * avg[0], 0.1)
        days = stock / demand

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Predicted sales next week", f"{demand:.1f} units/day")
        m2.metric("Stock will last about", f"{days:.1f} days")

        if days <= lead_time:
            st.error("CRITICAL REORDER WARNING: order stock now!")
        else:
            st.success("Stock level is OK.")


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
