from flask import Flask, render_template, request, jsonify
import numpy as np
import json

app = Flask(__name__)

# ─────────────────────────────────────────────
# FUZZY MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────

def trimf(x, a, b, c):
    """Triangular membership function"""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def trapmf(x, a, b, c, d):
    """Trapezoidal membership function"""
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)

# ─────────────────────────────────────────────
# INPUT: USIA (15–60 tahun)
# ─────────────────────────────────────────────
def usia_muda(x):      return trapmf(x, 15, 15, 20, 30)
def usia_dewasa(x):    return trimf(x, 20, 35, 50)
def usia_tua(x):       return trapmf(x, 40, 50, 60, 60)

# ─────────────────────────────────────────────
# INPUT: DETAK JANTUNG ISTIRAHAT / RHR (40–100 bpm)
# ─────────────────────────────────────────────
def rhr_rendah(x):     return trapmf(x, 40, 40, 55, 65)
def rhr_normal(x):     return trimf(x, 55, 68, 80)
def rhr_tinggi(x):     return trapmf(x, 72, 85, 100, 100)

# ─────────────────────────────────────────────
# INPUT: PENGALAMAN BOXING (0–10 tahun)
# ─────────────────────────────────────────────
def exp_pemula(x):     return trapmf(x, 0, 0, 1, 3)
def exp_menengah(x):   return trimf(x, 1, 4, 7)
def exp_ahli(x):       return trapmf(x, 5, 7, 10, 10)

# ─────────────────────────────────────────────
# OUTPUT: INTENSITAS LATIHAN (0–100%)
# ─────────────────────────────────────────────
output_range = np.linspace(0, 100, 1000)

def out_ringan(x):     return trapmf(x, 0, 0, 20, 40)
def out_sedang(x):     return trimf(x, 25, 50, 75)
def out_berat(x):      return trapmf(x, 60, 80, 100, 100)

# ─────────────────────────────────────────────
# MAMDANI FUZZY INFERENCE
# ─────────────────────────────────────────────

def fuzzy_inference(usia, rhr, pengalaman):
    # Fuzzifikasi
    u_muda    = usia_muda(usia)
    u_dewasa  = usia_dewasa(usia)
    u_tua     = usia_tua(usia)

    r_rendah  = rhr_rendah(rhr)
    r_normal  = rhr_normal(rhr)
    r_tinggi  = rhr_tinggi(rhr)

    e_pemula  = exp_pemula(pengalaman)
    e_menengah= exp_menengah(pengalaman)
    e_ahli    = exp_ahli(pengalaman)

    # ── RULE BASE (18 aturan Mamdani) ──────────────
    rules = []

    # Pemula → cenderung ringan-sedang
    rules.append(("ringan",  min(u_muda,   r_tinggi,  e_pemula)))   # R1
    rules.append(("ringan",  min(u_muda,   r_normal,  e_pemula)))   # R2
    rules.append(("ringan",  min(u_tua,    r_tinggi,  e_pemula)))   # R3
    rules.append(("ringan",  min(u_tua,    r_normal,  e_pemula)))   # R4
    rules.append(("sedang",  min(u_dewasa, r_normal,  e_pemula)))   # R5
    rules.append(("ringan",  min(u_dewasa, r_tinggi,  e_pemula)))   # R6

    # Menengah → sedang
    rules.append(("sedang",  min(u_muda,   r_rendah,  e_menengah))) # R7
    rules.append(("sedang",  min(u_muda,   r_normal,  e_menengah))) # R8
    rules.append(("sedang",  min(u_dewasa, r_rendah,  e_menengah))) # R9
    rules.append(("sedang",  min(u_dewasa, r_normal,  e_menengah))) # R10
    rules.append(("ringan",  min(u_tua,    r_tinggi,  e_menengah))) # R11
    rules.append(("sedang",  min(u_tua,    r_normal,  e_menengah))) # R12

    # Ahli → berat
    rules.append(("berat",   min(u_muda,   r_rendah,  e_ahli)))     # R13
    rules.append(("berat",   min(u_muda,   r_normal,  e_ahli)))     # R14
    rules.append(("berat",   min(u_dewasa, r_rendah,  e_ahli)))     # R15
    rules.append(("sedang",  min(u_dewasa, r_tinggi,  e_ahli)))     # R16
    rules.append(("sedang",  min(u_tua,    r_rendah,  e_ahli)))     # R17
    rules.append(("ringan",  min(u_tua,    r_tinggi,  e_ahli)))     # R18

    # Agregasi output (max-min)
    agg_ringan = np.zeros(len(output_range))
    agg_sedang = np.zeros(len(output_range))
    agg_berat  = np.zeros(len(output_range))

    for label, strength in rules:
        for i, x in enumerate(output_range):
            if label == "ringan":
                agg_ringan[i] = max(agg_ringan[i], min(strength, out_ringan(x)))
            elif label == "sedang":
                agg_sedang[i] = max(agg_sedang[i], min(strength, out_sedang(x)))
            elif label == "berat":
                agg_berat[i]  = max(agg_berat[i],  min(strength, out_berat(x)))

    agg_total = np.maximum(np.maximum(agg_ringan, agg_sedang), agg_berat)

    # Defuzzifikasi – Centroid (CoA)
    denom = np.sum(agg_total)
    if denom == 0:
        crisp = 50.0
    else:
        crisp = np.sum(output_range * agg_total) / denom

    # Membership values untuk visualisasi
    memberships = {
        "usia":       {"muda": round(u_muda,3), "dewasa": round(u_dewasa,3), "tua": round(u_tua,3)},
        "rhr":        {"rendah": round(r_rendah,3), "normal": round(r_normal,3), "tinggi": round(r_tinggi,3)},
        "pengalaman": {"pemula": round(e_pemula,3), "menengah": round(e_menengah,3), "ahli": round(e_ahli,3)},
    }

    # Tentukan label output
    if crisp < 33:
        label_out = "RINGAN"
        color = "#4ade80"
        icon = ""
        deskripsi = "Latihan ringan cocok untuk pemanasan, pemulihan, atau pemula. Fokus pada teknik dasar, footwork, dan shadow boxing."
        rekomendasi = ["Shadow Boxing 3 ronde × 3 menit", "Skipping tali 10 menit", "Teknik dasar jab & cross", "Latihan pernapasan", "Cool-down stretching 10 menit"]
    elif crisp < 66:
        label_out = "SEDANG"
        color = "#facc15"
        icon = ""
        deskripsi = "Latihan intensitas sedang untuk membangun daya tahan dan kekuatan teknik. Cocok untuk atlet berkembang."
        rekomendasi = ["Bag work 5 ronde × 3 menit", "Sparring ringan 3 ronde", "Kombinasi 1-2-3 intensif", "Jump rope interval 15 menit", "Core training 20 menit"]
    else:
        label_out = "BERAT"
        color = "#f87171"
        icon = ""
        deskripsi = "Latihan intensitas tinggi untuk atlet berpengalaman. Memaksimalkan kekuatan, kecepatan, dan kondisi pertandingan."
        rekomendasi = ["Full sparring 6 ronde × 3 menit", "Heavy bag power shots 5 ronde", "Speed bag + double-end bag", "Conditioning drill HIIT 20 menit", "Strength & power training"]

    # Data chart untuk output fuzzy
    chart_x = output_range[::10].tolist()
    chart_ringan = agg_ringan[::10].tolist()
    chart_sedang = agg_sedang[::10].tolist()
    chart_berat  = agg_berat[::10].tolist()
    chart_total  = agg_total[::10].tolist()

    fired_rules = [(r[0], round(r[1],3)) for r in rules if r[1] > 0.001]

    return {
        "crisp": round(float(crisp), 2),
        "label": label_out,
        "color": color,
        "icon": icon,
        "deskripsi": deskripsi,
        "rekomendasi": rekomendasi,
        "memberships": memberships,
        "fired_rules": fired_rules,
        "chart": {
            "x": chart_x,
            "ringan": chart_ringan,
            "sedang": chart_sedang,
            "berat":  chart_berat,
            "total":  chart_total,
        }
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hitung", methods=["POST"])
def hitung():
    data = request.get_json()
    usia       = float(data["usia"])
    rhr        = float(data["rhr"])
    pengalaman = float(data["pengalaman"])
    result = fuzzy_inference(usia, rhr, pengalaman)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
