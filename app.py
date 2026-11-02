import hashlib, io, json, re, secrets, sqlite3
import joblib, numpy as np, pandas as pd, streamlit as st
from features import make_features

st.set_page_config(page_title="Warehouse Manager", page_icon="📦", layout="wide")
DB_PATH = "users.db"
MODELS = {"Random Forest": "models/random_forest.pkl", "Decision Tree": "models/decision_tree.pkl",
          "Linear Regression": "models/linear_regression.pkl"}
SAFETY_DAYS = 7  # extra days of cover when suggesting a reorder quantity


# ============================== database
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, salt TEXT NOT NULL, pw_hash TEXT NOT NULL)")
    conn.execute("""CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, sku TEXT NOT NULL, name TEXT NOT NULL,
        unit_price REAL NOT NULL, stock INTEGER NOT NULL, lead_time INTEGER NOT NULL,
        s1 REAL NOT NULL DEFAULT 0, s2 REAL NOT NULL DEFAULT 0, s3 REAL NOT NULL DEFAULT 0, s4 REAL NOT NULL DEFAULT 0,
        updated TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(owner, sku))""")
    return conn


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def create_user(email, pw):
    salt = secrets.token_hex(16)
    try:
        with db() as c:
            c.execute("INSERT INTO users VALUES (?,?,?)", (email.lower(), salt, hash_pw(pw, salt)))
        return True
    except sqlite3.IntegrityError:
        return False


def check_user(email, pw):
    with db() as c:
        row = c.execute("SELECT salt, pw_hash FROM users WHERE email=?", (email.lower(),)).fetchone()
    return bool(row) and secrets.compare_digest(hash_pw(pw, row["salt"]), row["pw_hash"])


def get_products(owner):
    with db() as c:
        return pd.read_sql_query("SELECT * FROM products WHERE owner=? ORDER BY name", c, params=(owner,))


def add_product(owner, sku, name, price, stock, lead, sales):
    try:
        with db() as c:
            c.execute("INSERT INTO products (owner,sku,name,unit_price,stock,lead_time,s1,s2,s3,s4) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (owner, sku.strip(), name.strip(), price, stock, lead, *sales))
        return True
    except sqlite3.IntegrityError:
        return False


def update_product(owner, pid, **f):
    cols = ", ".join(f"{k}=?" for k in f)
    with db() as c:
        c.execute(f"UPDATE products SET {cols}, updated=CURRENT_TIMESTAMP WHERE id=? AND owner=?", (*f.values(), pid, owner))


def delete_products(owner, ids):
    with db() as c:
        c.executemany("DELETE FROM products WHERE id=? AND owner=?", [(int(i), owner) for i in ids])


# ============================== model + forecasting
@st.cache_resource
def load_model(name):
    return joblib.load(MODELS[name])


@st.cache_data
def load_metrics():
    try:
        return json.load(open("models/metrics.json"))
    except FileNotFoundError:
        return None


def forecast(df, model_name):
    """Adds predicted demand, days of stock left, status and suggested reorder qty to a products frame."""
    if df.empty:
        return df
    out = df.copy()
    lags = out[["s1", "s2", "s3", "s4"]].values
    X, avg = make_features(lags, out["unit_price"], np.zeros(len(out)))
    X["is_festival_season"] = int(st.session_state.get("festival", False))
    ratio = np.clip(load_model(model_name).predict(X), 0, None)
    out["demand"] = np.where(avg == 0, 0.1, np.maximum(ratio * avg, 0.1)).round(2)
    out["days_left"] = (out["stock"] / out["demand"]).round(1)
    out["status"] = np.select([out["days_left"] <= out["lead_time"], out["days_left"] <= out["lead_time"] * 1.5],
                              ["🔴 Reorder now", "🟡 Reorder soon"], "🟢 OK")
    need = out["demand"] * (out["lead_time"] + SAFETY_DAYS) - out["stock"]
    out["reorder_qty"] = np.where(out["days_left"] <= out["lead_time"] * 1.5, np.ceil(need).clip(min=0), 0).astype(int)
    return out


# ============================== auth
if "user" not in st.session_state:
    st.session_state.user = None


def auth_page():
    st.title("📦 Warehouse Manager")
    st.caption("Manage products, forecast next week's demand and get reorder warnings.")
    t1, t2 = st.tabs(["Login", "Sign Up"])
    with t1, st.form("login"):
        e, p = st.text_input("Email"), st.text_input("Password", type="password")
        if st.form_submit_button("Login", use_container_width=True):
            if check_user(e.strip(), p):
                st.session_state.user = e.strip().lower(); st.rerun()
            st.error("Incorrect email or password.")
    with t2, st.form("signup"):
        ne = st.text_input("Email", key="se")
        npw = st.text_input("Password", type="password", key="sp")
        cf = st.text_input("Confirm password", type="password")
        if st.form_submit_button("Create account", use_container_width=True):
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", ne.strip()): st.error("Enter a valid email address.")
            elif len(npw) < 6: st.error("Password must be at least 6 characters.")
            elif npw != cf: st.error("Passwords don't match.")
            elif create_user(ne.strip(), npw): st.success("Account created. Open the Login tab.")
            else: st.error("An account with this email already exists.")


# ============================== pages
def manage(user):
    st.title("Manage Products")
    tab_add, tab_edit, tab_sales, tab_bulk = st.tabs(["➕ Add", "✏️ Edit / Delete", "📈 Record weekly sales", "📁 Bulk import"])
    df = get_products(user)

    with tab_add, st.form("add", clear_on_submit=True):
        a, b = st.columns(2)
        sku, name = a.text_input("SKU / product code"), b.text_input("Product name")
        c1, c2, c3 = st.columns(3)
        price = c1.number_input("Unit price", 0.1, 100000.0, 2.0)
        stock = c2.number_input("Current stock", 0, 10_000_000, 100)
        lead = c3.number_input("Lead time (days)", 1, 90, 7)
        st.write("Past sales (units/day)")
        w = st.columns(4)
        sales = [w[i].number_input(l, 0.0, 100000.0, 10.0, key=f"a{i}") for i, l in enumerate(["Last week", "2 weeks ago", "3 weeks ago", "4 weeks ago"])]
        if st.form_submit_button("Add product", use_container_width=True):
            if not sku.strip() or not name.strip(): st.error("SKU and name are required.")
            elif add_product(user, sku, name, price, stock, lead, sales): st.success(f"Added {name}."); st.rerun()
            else: st.error("That SKU already exists.")

    with tab_edit:
        if df.empty: st.info("Nothing to edit yet.")
        else:
            label = {r.id: f"{r.sku} - {r.name}" for r in df.itertuples()}
            pid = st.selectbox("Product", list(label), format_func=label.get)
            r = df[df.id == pid].iloc[0]
            with st.form("edit"):
                name = st.text_input("Name", r["name"])
                c1, c2, c3 = st.columns(3)
                price = c1.number_input("Unit price", 0.1, 100000.0, float(r.unit_price))
                stock = c2.number_input("Current stock", 0, 10_000_000, int(r.stock))
                lead = c3.number_input("Lead time (days)", 1, 90, int(r.lead_time))
                if st.form_submit_button("Save changes", use_container_width=True):
                    update_product(user, int(pid), name=name, unit_price=price, stock=stock, lead_time=lead)
                    st.success("Saved."); st.rerun()
            q = st.number_input("Received shipment (units to add to stock)", 0, 10_000_000, 0)
            if st.button("Add received stock") and q:
                update_product(user, int(pid), stock=int(r.stock) + int(q)); st.rerun()
            if st.checkbox(f"Confirm delete {label[pid]}") and st.button("Delete product", type="primary"):
                delete_products(user, [pid]); st.rerun()

    with tab_sales:
        st.caption("At the end of each week enter units sold per day. It becomes 'last week', older weeks shift back, and stock is reduced by the units sold.")
        if df.empty: st.info("Add products first.")
        else:
            entry = pd.DataFrame({"id": df.id, "Product": df.name, "Units sold per day (this week)": 0.0})
            ed = st.data_editor(entry, hide_index=True, disabled=["id", "Product"], use_container_width=True, key="wk")
            if st.button("Save week and shift history", type="primary"):
                for r, row in ed.iterrows():
                    p = df[df.id == row.id].iloc[0]; v = float(row.iloc[2])
                    update_product(user, int(p.id), s1=v, s2=p.s1, s3=p.s2, s4=p.s3, stock=max(0, int(p.stock - v * 7)))
                st.success("Weekly sales recorded."); st.rerun()

    with tab_bulk:
        st.caption("CSV columns: sku, name, unit_price, stock, lead_time, s1, s2, s3, s4 (existing SKUs are updated).")
        tpl = pd.DataFrame([["A100", "Mug", 3.5, 200, 7, 12, 10, 9, 11]], columns=["sku", "name", "unit_price", "stock", "lead_time", "s1", "s2", "s3", "s4"])
        st.download_button("Download template", tpl.to_csv(index=False), "products_template.csv")
        up = st.file_uploader("Upload CSV", type="csv")
        if up and st.button("Import"):
            try:
                imp = pd.read_csv(up); n = 0
                for r in imp.itertuples():
                    if not add_product(user, str(r.sku), r.name, r.unit_price, int(r.stock), int(r.lead_time), [r.s1, r.s2, r.s3, r.s4]):
                        pid = df.loc[df.sku == str(r.sku), "id"].iloc[0]
                        update_product(user, int(pid), name=r.name, unit_price=r.unit_price, stock=int(r.stock), lead_time=int(r.lead_time), s1=r.s1, s2=r.s2, s3=r.s3, s4=r.s4)
                    n += 1
                st.success(f"Imported {n} products."); st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")


def comparison():
    st.title("Model Comparison")
    m = load_metrics()
    if not m: st.warning("Run `python train_models.py` first."); return
    t = pd.DataFrame(m["results"]).T[["MAE", "RMSE", "R2", "CV_R2_ratio", "best_params"]]
    st.dataframe(t, use_container_width=True)
    st.success(f"Best model on the test set (lowest MAE): **{m['best_model']}**")
    st.caption(f"Evaluated on {m['n_test']} held-out rows. MAE/RMSE are in units per day; lower is better. R2: higher is better.")
    for f, cap in [("model_comparison", "Metric comparison"), ("actual_vs_predicted", "Actual vs predicted"), ("feature_importance", "Random Forest feature importance")]:
        try: st.image(f"models/{f}.png", caption=cap)
        except Exception: pass


# ============================== router
if not st.session_state.user:
    auth_page()
else:
    user = st.session_state.user
    with st.sidebar:
        st.write(f"Signed in as **{user}**")
        page = st.radio("Page", ["Manage Products", "Model Comparison"])
        if st.button("Log out", use_container_width=True):
            st.session_state.user = None; st.rerun()
    {"Manage Products": manage}.get(page, lambda u: comparison())(user)
