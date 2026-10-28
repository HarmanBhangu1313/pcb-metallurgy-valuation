import argparse
import os
import glob
import json
from collections import Counter
import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def detect():
    parser = argparse.ArgumentParser(description="Run PCB Component Detection & Metallurgical Profit Valuation (INR ₹)")
    parser.add_argument("--source", type=str, default=os.path.join(PROJECT_ROOT, "dataset/test/images"), help="Path to image or image directory")
    parser.add_argument("--weights", type=str, default=os.path.join(PROJECT_ROOT, "models/pcb_best_model.pt"), help="Path to weights file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Device: 'mps', 'cpu', '0'")
    parser.add_argument("--save-dir", type=str, default=os.path.join(PROJECT_ROOT, "runs/detect/output"), help="Output directory")
    parser.add_argument("--max-images", type=int, default=10, help="Maximum number of images to process (0 for all)")
    parser.add_argument("--metallurgy", action="store_true", default=True, help="Compute metallurgical elemental recovery & valuation")
    parser.add_argument("--no-live", action="store_true", help="Disable live price fetching, use offline benchmark")
    parser.add_argument("--gold-key", type=str, default="", help="Optional GoldAPI.io free API key")
    parser.add_argument("--export-json", type=str, default="", help="Path to export JSON valuation report")
    args = parser.parse_args()

    # Resolve absolute path for weights
    weights = args.weights
    if not os.path.exists(weights):
        alt_weights = os.path.join(PROJECT_ROOT, "models", "pcb_best_model.pt")
        if os.path.exists(alt_weights):
            weights = alt_weights
        else:
            print(f"⚠️ Warning: Checkpoint '{weights}' not found. Using yolov8s.pt fallback.")
            weights = "yolov8s.pt"

    if not args.device:
        device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    model = YOLO(weights)
    os.makedirs(args.save_dir, exist_ok=True)

    # Gather image files
    if os.path.isdir(args.source):
        img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        images = []
        for ext in img_extensions:
            images.extend(glob.glob(os.path.join(args.source, ext)))
        if args.max_images > 0:
            images = images[:args.max_images]
    else:
        images = [args.source]

    if not images:
        print(f"❌ No images found at source: {args.source}")
        return

    print("=" * 72)
    print("🔍 MACHINE VISION PCB DETECTION & METALLURGICAL VALUATION (INR ₹)")
    print("   Department of Metallurgical Engineering, IIT (BHU) Varanasi")
    print("=" * 72)
    print(f"   Model Weights : {weights}")
    print(f"   Target Images : {len(images)}")
    print(f"   Confidence    : {args.conf} | IoU: {args.iou} | Device: {device}")
    print(f"   Save Directory: {args.save_dir}")
    print("=" * 72)

    total_counts = Counter()

    for idx, img_path in enumerate(images, 1):
        filename = os.path.basename(img_path)
        results = model.predict(
            source=img_path,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )

        result = results[0]
        img_counts = Counter()
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            img_counts[cls_name] += 1
            total_counts[cls_name] += 1

        output_path = os.path.join(args.save_dir, f"detected_{filename}")
        annotated_frame = result.plot()
        cv2.imwrite(output_path, annotated_frame)

        print(f"\n[{idx}/{len(images)}] {filename}:")
        if img_counts:
            for cls_name, count in img_counts.most_common():
                print(f"   • {cls_name:<16}: {count} detected")
        else:
            print("   • No components detected above confidence threshold.")
        print(f"   Saved annotation to: {output_path}")

    print("\n" + "=" * 72)
    print("📊 AGGREGATE COMPONENT INVENTORY ACROSS EVALUATED BOARDS:")
    print("=" * 72)
    for cls_name, count in total_counts.most_common():
        print(f"   • {cls_name:<20}: {count:>4} instances")
    print(f"   {'TOTAL DETECTIONS':<20}: {sum(total_counts.values()):>4} instances")
    print("=" * 72)

    if args.metallurgy and total_counts:
        try:
            from metallurgy_engine import calculate_metallurgical_yield, get_component_range_profile
            rep = calculate_metallurgical_yield(
                total_counts,
                use_live_prices=not args.no_live,
                api_key=args.gold_key
            )
            sm = rep["summary"]
            print("\n" + "=" * 72)
            print("⚗️  METALLURGICAL ASSAY & PROFIT RANGE ANALYSIS (INR ₹)")
            print("=" * 72)
            print(f"📊 Market Pricing Feed : {sm['market_feed_status']} ({sm['market_feed_source']})")
            print(f"⏰ Feed Timestamp      : {sm['market_feed_timestamp']}")
            print(f"⚖️ Component Mass Range: {sm['component_mass_g_range']['min']}g (Min) | {sm['component_mass_g_range']['avg']}g (Avg) | {sm['component_mass_g_range']['max']}g (Max)")
            print("─" * 72)
            print("💎 RECOVERABLE METALS & REVENUE RANGE (₹ INR):")
            print(f"{'Metal':<16} {'Mass Range (mg/g)':<28} {'Gross Revenue (INR)':<24}")
            print("─" * 72)
            for m, d in rep["metals"].items():
                mass_str = f"{d['recoverable_mass_mg_range']['min']} - {d['recoverable_mass_mg_range']['avg']} - {d['recoverable_mass_mg_range']['max']} mg"
                val_str = f"₹{d['gross_revenue_inr_range']['min']:,.2f} - ₹{d['gross_revenue_inr_range']['avg']:,.2f} - ₹{d['gross_revenue_inr_range']['max']:,.2f}"
                print(f"• {m:<14}: {mass_str:<28} -> {val_str}")
            print("─" * 72)

            gv_i = rep["gross_valuation_inr"]
            ec_i = rep["extraction_cost_inr"]
            np_i = rep["net_profit_inr"]
            pm = rep["net_profit_margin_percent"]

            print("💰 FINANCIAL SUMMARY (RUPEES ONLY):")
            print(f"   • Gross Metal Value : ₹{gv_i['min']:,.2f} – ₹{gv_i['avg']:,.2f} – ₹{gv_i['max']:,.2f}")
            print(f"   • Extraction Cost   : ₹{ec_i['min']:,.2f} – ₹{ec_i['avg']:,.2f} – ₹{ec_i['max']:,.2f}  (Acids, energy & sorting)")
            print("   --------------------------------------------------------------------")
            print(f"   🚀 NET PROFIT RANGE : ₹{np_i['min']:,.2f} – ₹{np_i['avg']:,.2f} – ₹{np_i['max']:,.2f} INR")
            print(f"   📈 Net Profit Margin: {pm['min']}% – {pm['avg']}% – {pm['max']}%")
            print("─" * 72)

            if total_counts.get("IC", 0) > 0:
                print("\n🔍 SPOTLIGHT: GOLD (Au) RECOVERY FROM DETECTED ICs (₹ INR):")
                ic_prof = get_component_range_profile("IC", "Au", count=total_counts["IC"])
                au = ic_prof["elements"]["Au"]
                print(f"   • Detected IC Count : {total_counts['IC']} units")
                print(f"   • Gold Mass Range   : {au['total_mass_mg']['min']} mg (Min) | {au['total_mass_mg']['avg']} mg (Avg) | {au['total_mass_mg']['max']} mg (Max)")
                print(f"   • Recoverable Pure  : {au['recoverable_mass_mg']['min']} mg (Min) | {au['recoverable_mass_mg']['avg']} mg (Avg) | {au['recoverable_mass_mg']['max']} mg (Max)")
                print(f"   • Net Financial Gain: ₹{au['recoverable_value_inr']['min']:,.2f} – ₹{au['recoverable_value_inr']['avg']:,.2f} – ₹{au['recoverable_value_inr']['max']:,.2f}")
                print("─" * 72)

            rq = rep["reagents_stoichiometry"]
            print("\n🧪 REQUIRED LEACHING REAGENTS (STOHIOMETRIC MASS BALANCE):")
            print(f"   • Conc. HNO3 (68%)  : {rq['nitric_acid_68pct_ml_range']['min']} – {rq['nitric_acid_68pct_ml_range']['avg']} – {rq['nitric_acid_68pct_ml_range']['max']} mL")
            print(f"   • Aqua Regia Soln   : {rq['aqua_regia_ml_range']['min']} – {rq['aqua_regia_ml_range']['avg']} – {rq['aqua_regia_ml_range']['max']} mL")
            print(f"   • Na2S2O5 Powder    : {rq['na2s2o5_precipitant_g_avg']:.3f} grams (Precipitant)")
            print("=" * 72)

            if args.export_json:
                with open(args.export_json, "w") as f:
                    json.dump(rep, f, indent=2)
                print(f"\n💾 Exported detailed valuation report to: {args.export_json}")

        except Exception as e:
            print(f"⚠️ Could not compute metallurgy metrics: {e}")

if __name__ == "__main__":
    detect()
