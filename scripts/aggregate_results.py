import os
import json
import glob
import csv

EXPERIMENTS_DIR = '/home/ML_courses/03683533_2025/ameer_george_abdallah/experiments/runs'
OUTPUT_DIR = '/home/ML_courses/03683533_2025/ameer_george_abdallah/report_assets'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_metrics(run_dir):
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)

def create_svg_chart(run_id, metrics, output_path):
    width = 800
    height = 600
    padding = 50
    
    train_loss = metrics.get('train_l1', [])
    val_loss = metrics.get('val_l1', [])
    
    if not train_loss:
        return

    epochs = len(train_loss)
    max_val = max(max(train_loss), max(val_loss) if val_loss else 0)
    min_val = min(min(train_loss), min(val_loss) if val_loss else 0)
    
    # Normalize to canvas
    def x_scale(epoch_idx):
        return padding + (epoch_idx / (epochs - 1)) * (width - 2 * padding)
    
    def y_scale(val):
        return height - padding - ((val - min_val) / (max_val - min_val if max_val != min_val else 1)) * (height - 2 * padding)
    
    svg_content = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    
    # Background
    svg_content.append(f'<rect width="100%" height="100%" fill="white"/>')
    
    # Axes
    svg_content.append(f'<line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="black"/>') # X axis
    svg_content.append(f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="black"/>') # Y axis
    
    # Title
    svg_content.append(f'<text x="{width/2}" y="{padding/2}" text-anchor="middle" font-family="sans-serif" font-size="20">{run_id}</text>')
    
    # Plot Train Loss (Blue)
    points = []
    for i, val in enumerate(train_loss):
        points.append(f'{x_scale(i)},{y_scale(val)}')
    svg_content.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="blue" stroke-width="2"/>')
    
    # Plot Val Loss (Red)
    if val_loss:
        points = []
        for i, val in enumerate(val_loss):
            points.append(f'{x_scale(i)},{y_scale(val)}')
        svg_content.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="red" stroke-width="2"/>')
        
    # Legend
    svg_content.append(f'<rect x="{width-150}" y="{padding}" width="100" height="50" fill="white" stroke="black"/>')
    svg_content.append(f'<line x1="{width-140}" y1="{padding+15}" x2="{width-120}" y2="{padding+15}" stroke="blue" stroke-width="2"/>')
    svg_content.append(f'<text x="{width-115}" y="{padding+20}" font-family="sans-serif" font-size="12">Train L1</text>')
    if val_loss:
        svg_content.append(f'<line x1="{width-140}" y1="{padding+35}" x2="{width-120}" y2="{padding+35}" stroke="red" stroke-width="2"/>')
        svg_content.append(f'<text x="{width-115}" y="{padding+40}" font-family="sans-serif" font-size="12">Val L1</text>')

    svg_content.append('</svg>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_content))

def aggregate_results():
    run_dirs = glob.glob(os.path.join(EXPERIMENTS_DIR, '*'))
    results = []
    
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        if not os.path.isdir(run_dir):
            continue
            
        metrics = load_metrics(run_dir)
        
        if metrics is None or 'train_l1' not in metrics:
            print(f"Skipping {run_id}: No metrics found.")
            continue
            
        # Get final metrics
        final_train_l1 = metrics['train_l1'][-1]
        final_val_l1 = metrics['val_l1'][-1] if 'val_l1' in metrics else None
        final_val_psnr = metrics['val_psnr'][-1] if 'val_psnr' in metrics else None
        
        # Determine model type and details from run_id (heuristic)
        is_vae = 'vae' in run_id
        is_attention = 'attention' in run_id
        has_lpips = 'lpips' in run_id
        
        results.append({
            'Run ID': run_id,
            'Type': 'VAE' if is_vae else 'AE',
            'Attention': is_attention,
            'LPIPS': has_lpips,
            'Final Train L1': final_train_l1,
            'Final Val L1': final_val_l1,
            'Final Val PSNR': final_val_psnr
        })
        
        # Generate plot
        plot_path = os.path.join(OUTPUT_DIR, f'{run_id}_loss.svg')
        create_svg_chart(run_id, metrics, plot_path)
        print(f"Generated plot for {run_id} at {plot_path}")

    # Sort by Validation L1 Loss (ascending)
    # Handle None values safely
    results.sort(key=lambda x: x['Final Val L1'] if x['Final Val L1'] is not None else float('inf'))
        
    # Save summary CSV
    csv_path = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Run ID', 'Type', 'Attention', 'LPIPS', 'Final Train L1', 'Final Val L1', 'Final Val PSNR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"Saved summary to {csv_path}")
    
    # Identify best model
    if results:
        best_model = results[0]
        print(f"Best Model: {best_model['Run ID']} with Val L1: {best_model['Final Val L1']}")
        
        # Save best model info
        with open(os.path.join(OUTPUT_DIR, 'best_model.json'), 'w') as f:
            json.dump(best_model, f, indent=4)

if __name__ == '__main__':
    aggregate_results()
