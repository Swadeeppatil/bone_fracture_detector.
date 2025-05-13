import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import sqlite3
from datetime import datetime
import os

class BoneFractureDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bone Fracture Detection System")
        self.root.geometry("1400x800")
        
        # Initialize fracture types first
        self.fracture_types = {
            "transverse": {
                "description": "Clean break across the bone",
                "severity": "Moderate",
                "treatment": "1. Immobilization\n2. Cast application\n3. Regular monitoring"
            },
            "oblique": {
                "description": "Angled break across the bone",
                "severity": "Moderate to Severe",
                "treatment": "1. Surgical evaluation\n2. Possible internal fixation\n3. Physical therapy"
            },
            "compound": {
                "description": "Bone pierces through skin",
                "severity": "Severe",
                "treatment": "1. Immediate surgery\n2. Antibiotics\n3. Wound care"
            },
            "stress": {
                "description": "Tiny cracks in bone",
                "severity": "Mild to Moderate",
                "treatment": "1. Rest and immobilization\n2. Activity modification\n3. Physical therapy"
            }
        }
        
        # Then initialize other components
        self.init_database()
        self.model = self.load_model()
        self.setup_gui()

    def init_database(self):
        self.conn = sqlite3.connect('fracture_scans.db')
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS scans
                    (id INTEGER PRIMARY KEY,
                     fracture_type TEXT,
                     confidence REAL,
                     severity TEXT,
                     timestamp DATETIME,
                     image_path TEXT)''')
        self.conn.commit()

    def load_model(self):
        # Using DenseNet169 for better feature extraction
        base_model = DenseNet169(weights='imagenet', include_top=False, 
                               input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(len(self.fracture_types), 
                                         activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        return model

    def setup_gui(self):
        # Create main frames
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH)
        
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        self.setup_image_upload()
        self.setup_analysis_section()
        self.setup_history_section()

    def setup_image_upload(self):
        upload_frame = ttk.LabelFrame(self.left_frame, text="X-Ray Image Upload")
        upload_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Image preview
        self.image_label = ttk.Label(upload_frame)
        self.image_label.pack(pady=10)
        
        # Control buttons
        btn_frame = ttk.Frame(upload_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Upload X-Ray", 
                  command=self.upload_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Enhance Image", 
                  command=self.enhance_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Detect Fracture", 
                  command=self.analyze_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Image", 
                  command=self.clear_image).pack(side=tk.LEFT, padx=5)

    def setup_history_section(self):
        history_frame = ttk.LabelFrame(self.left_frame, text="Scan History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frame for treeview and scrollbars
        tree_frame = ttk.Frame(history_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add both vertical and horizontal scrollbars
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        x_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        
        # History tree with scrollbars
        self.history_tree = ttk.Treeview(tree_frame, 
                                       columns=("Date", "Type", "Severity"),
                                       show="headings",
                                       yscrollcommand=y_scrollbar.set,
                                       xscrollcommand=x_scrollbar.set)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.history_tree.yview)
        x_scrollbar.config(command=self.history_tree.xview)
        
        # Pack scrollbars and treeview
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure column headings
        self.history_tree.heading("Date", text="Date")
        self.history_tree.heading("Type", text="Fracture Type")
        self.history_tree.heading("Severity", text="Severity")
        
        # Configure column widths
        self.history_tree.column("Date", width=150)
        self.history_tree.column("Type", width=150)
        self.history_tree.column("Severity", width=150)

    def clear_image(self):
        # Clear the image label
        self.image_label.configure(image='')
        self.image_label.image = None
        
        # Clear the current image path if it exists
        if hasattr(self, 'current_image_path'):
            delattr(self, 'current_image_path')
        
        # Clear the results
        self.results_text.delete(1.0, tk.END)
        self.detail_text.delete(1.0, tk.END)
        
        messagebox.showinfo("Success", "Image cleared successfully")

    def setup_analysis_section(self):
        analysis_frame = ttk.LabelFrame(self.right_frame, text="Analysis Results")
        analysis_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Results display
        self.results_text = tk.Text(analysis_frame, height=10, width=60, 
                                  font=('Arial', 12))
        self.results_text.pack(pady=10, padx=5)
        
        # Detailed analysis
        self.detail_text = tk.Text(analysis_frame, height=15, width=60, 
                                 font=('Arial', 12))
        self.detail_text.pack(pady=10, padx=5)

    def setup_history_section(self):
        history_frame = ttk.LabelFrame(self.left_frame, text="Scan History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frame for treeview and scrollbar
        tree_frame = ttk.Frame(history_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # History tree with scrollbar
        self.history_tree = ttk.Treeview(tree_frame, 
                                       columns=("Date", "Type", "Severity"),
                                       show="headings")
        self.history_tree.heading("Date", text="Date")
        self.history_tree.heading("Type", text="Fracture Type")
        self.history_tree.heading("Severity", text="Severity")
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, 
                                command=self.history_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_tree.configure(yscrollcommand=scrollbar.set)

    def enhance_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please upload an image first")
            return
            
        # Read image
        img = cv2.imread(self.current_image_path, 0)  # Read as grayscale
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # Apply additional enhancement
        enhanced = cv2.GaussianBlur(enhanced, (5,5), 0)
        enhanced = cv2.addWeighted(img, 1.5, enhanced, -0.5, 0)
        
        # Save enhanced image
        enhanced_path = self.current_image_path.replace('.', '_enhanced.')
        cv2.imwrite(enhanced_path, enhanced)
        
        # Update display
        self.current_image_path = enhanced_path
        self.display_image(enhanced_path)
        
        messagebox.showinfo("Success", "Image enhanced successfully")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.dcm")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)

    def display_image(self, path):
        image = Image.open(path)
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def analyze_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("Error", "Please upload an image first")
            return
        
        # Prepare image
        img = image.load_img(self.current_image_path, 
                           target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Get predictions
        preds = self.model.predict(x)
        fracture_type = list(self.fracture_types.keys())[np.argmax(preds[0])]
        confidence = np.max(preds[0]) * 100
        
        # Highlight fracture area
        self.highlight_fracture_area()
        
        # Display results
        self.display_results(fracture_type, confidence)
        
        # Save to database
        self.save_scan_result(fracture_type, confidence)
        
        # Update history
        self.update_history()

    def highlight_fracture_area(self):
        # Read the image
        img = cv2.imread(self.current_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the bone)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Find the area with highest intensity variation (potential fracture)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, (255), -1)
            roi = cv2.bitwise_and(gray, mask)
            
            # Apply local variance to detect anomalies
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            local_var = cv2.dilate(roi, kernel) - cv2.erode(roi, kernel)
            
            # Find point of maximum variation
            _, _, _, max_loc = cv2.minMaxLoc(local_var)
            
            # Draw circle around potential fracture
            cv2.circle(img, max_loc, 30, (0, 0, 255), 2)
            
            # Save highlighted image
            highlighted_path = self.current_image_path.replace('.', '_highlighted.')
            cv2.imwrite(highlighted_path, img)
            
            # Update display
            self.display_image(highlighted_path)

    def display_results(self, fracture_type, confidence):
        self.results_text.delete(1.0, tk.END)
        self.detail_text.delete(1.0, tk.END)
        
        # Basic results
        results = f"Analysis Results:\n\n"
        results += f"Detected Fracture Type: {fracture_type.title()}\n"
        results += f"Confidence: {confidence:.2f}%\n"
        self.results_text.insert(tk.END, results)
        
        # Detailed analysis
        if fracture_type in self.fracture_types:
            info = self.fracture_types[fracture_type]
            details = f"\nDetailed Analysis:\n\n"
            details += f"Description:\n{info['description']}\n\n"
            details += f"Severity Level:\n{info['severity']}\n\n"
            details += f"Recommended Treatment:\n{info['treatment']}\n\n"
            details += "\nAdditional Notes:\n"
            details += "• Consult with orthopedic specialist\n"
            details += "• Follow-up X-rays recommended\n"
            details += "• Monitor healing progress\n"
            self.detail_text.insert(tk.END, details)

    def save_scan_result(self, fracture_type, confidence):
        c = self.conn.cursor()
        c.execute("""INSERT INTO scans 
                    (fracture_type, confidence, severity, timestamp, image_path) 
                    VALUES (?, ?, ?, ?, ?)""",
                 (fracture_type, 
                  confidence,
                  self.fracture_types[fracture_type]['severity'],
                  datetime.now(),
                  self.current_image_path))
        self.conn.commit()

    def update_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        c = self.conn.cursor()
        c.execute("""SELECT timestamp, fracture_type, severity 
                    FROM scans 
                    ORDER BY timestamp DESC 
                    LIMIT 10""")
        
        for scan in c.fetchall():
            self.history_tree.insert('', tk.END, values=scan)

    def run(self):
        self.root.mainloop()
        self.conn.close()

if __name__ == "__main__":
    app = BoneFractureDetector()
    app.run()