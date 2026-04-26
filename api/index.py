from flask import Flask, render_template, request, jsonify
import numpy as np
import json

app = Flask(__name__, template_folder='../templates')

def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    elif a < x <= b: return (x - a) / (b - a)
    else: return (c - x) / (c - b)

def trapmf(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    elif a < x <= b: return (x - a) / (b - a)
    elif b < x <= c: return 1.0
    else: return (d - x) / (d - c)

# Input Functions
def usia_muda(x):      return trapmf(x, 15, 15, 20, 30)
def usia_dewasa(x):    return trimf(x, 20, 35, 50)
def usia_tua(x):       return trapmf(x, 40, 50, 60, 60)

def rhr_rendah(x):     return trapmf(x, 40, 40, 55, 65)
def rhr_normal(x):     return trimf(x, 55, 68, 80)
def rhr_tinggi(x):     return trapmf(x, 72, 85, 100, 100)

def exp_pemula(x):     return trapmf(x, 0, 0, 1, 3)
def exp_menengah(x):   return trimf(x, 1, 4, 7)
def exp_ahli(x):       return trapmf(x, 5, 7, 10, 10)

output_range = np.linspace(0, 100, 1000)

def out_ringan(x):     return trapmf(x, 0, 0, 20, 40)
def out_sedang(x):     return trimf(x, 25, 50, 75)
def out_berat(x):      return trapmf(x, 60, 80, 100, 100)

def fuzzy_inference(usia, rhr, pengalaman):
    # Fuzzifikasi
    u_muda, u_dewasa, u_tua = usia_muda(usia), usia_dewasa(usia), usia_tua(usia)
    r_rendah, r_normal, r_tinggi = rhr_rendah(rhr), rhr_normal(rhr), rhr_tinggi(rhr)
    e_pemula, e_menengah, e_ahli = exp_pemula(pengalaman), exp_menengah(pengalaman), exp_ahli(pengalaman)

    rules = []
    # Rule Base
    rules.append(("ringan", min(u_muda, r_tinggi, e_pemula)))
    rules.append(("ringan", min(u_muda, r_normal, e_pemula)))
    rules.append(("ringan", min(u_tua, r_tinggi, e_pemula)))
    rules.append(("ringan", min(u_tua, r_normal, e_pemula)))
    rules.append(("sedang", min(u_dewasa, r_normal, e_pemula)))
    rules.append(("ringan", min(u_dewasa, r_tinggi, e_pemula)))
    rules.append(("sedang", min(u_muda, r_rendah, e_menengah)))
    rules.append(("sedang", min(u_muda, r_normal, e_menengah)))
    rules.append(("sedang", min(u_dewasa, r_rendah, e_menengah)))
    rules.append(("sedang", min(u_dewasa, r_normal, e_menengah)))
    rules.append(("ringan", min(u_tua, r_tinggi, e_menengah)))
    rules.append(("sedang", min(u_tua, r_normal, e_menengah)))
    rules.append(("berat",  min(u_muda, r_rendah, e_ahli)))
    rules.append(("berat",  min(u_muda, r_normal, e_ahli)))
    rules.append(("berat",  min(u_dewasa, r_rendah, e_ahli)))
    rules.append(("sedang", min(u_dewasa, r_tinggi, e_ahli)))
    rules.append(("sedang", min(u_tua, r_rendah, e_ahli)))
    rules.append(("ringan", min(u_tua, r_tinggi, e_ahli)))

    agg_ringan = np.zeros(len(output_range))
    agg_sedang = np.zeros(len(output_range))
    agg_berat  = np.zeros(len(output_range))

    for label, strength in rules:
        for i, x in enumerate(output_range):
            if label == "ringan": agg_ringan[i] = max(agg_ringan[i], min(strength, out_ringan(x)))
            elif label == "sedang": agg_sedang[i] = max(agg_sedang[i], min(strength, out_sedang(x)))
            elif label == "berat": agg_berat[i]  = max(agg_berat[i],  min(strength, out_berat(x)))

    agg_total = np.maximum(np.maximum(agg_ringan, agg_sedang), agg_berat)

    denom = np.sum(agg_total)
    crisp = 50.0 if denom == 0 else np.sum(output_range * agg_total) / denom

    # Penentuan Label, Color, Icon & Rekomendasi
    if crisp < 33:
        label_out, color, icon = "RINGAN", "#4ade80", "🟢"
        deskripsi = "Latihan ringan cocok untuk pemanasan atau pemulihan."
        rekomendasi = ["Shadow Boxing 3 ronde", "Skipping 10 menit", "Teknik dasar"]
    elif crisp < 66:
        label_out, color, icon = "SEDANG", "#facc15", "🟡"
        deskripsi = "Latihan intensitas sedang untuk membangun daya tahan."
        rekomendasi = ["Bag work 5 ronde", "Sparring ringan", "Core training"]
    else:
        label_out, color, icon = "BERAT", "#f87171", "🔴"
        deskripsi = "Latihan intensitas tinggi untuk atlet berpengalaman."
        rekomendasi = ["Full sparring", "Heavy bag power", "HIIT conditioning"]

    return {
        "crisp": round(float(crisp), 2),
        "label": label_out,
        "color": color,
        "icon": icon,
        "deskripsi": deskripsi,
        "rekomendasi": rekomendasi,
        "chart": {
            "x": output_range[::10].tolist(),
            "total": agg_total[::10].tolist(),
            "ringan": agg_ringan[::10].tolist(),
            "sedang": agg_sedang[::10].tolist(),
            "berat": agg_berat[::10].tolist()
        }
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hitung", methods=["POST"])
def hitung():
    data = request.get_json()
    try:
        usia = float(data.get("usia", 0))
        rhr = float(data.get("rhr", 0))
        pengalaman = float(data.get("pengalaman", 0))
        result = fuzzy_inference(usia, rhr, pengalaman)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)