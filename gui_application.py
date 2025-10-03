"""
Main GUI Application using Tkinter with OOP concepts
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import logging
from image_classification_model import ImageClassificationModel
from sentiment_analysis_model import SentimentAnalysisModel
import os

class ModelFactory:
    """Factory pattern for creating models - demonstrates design patterns"""
    
    @staticmethod
    def create_model(model_type: str):
        """Factory method to create models"""
        if model_type == "Image Classification":
            return ImageClassificationModel()
        elif model_type == "Sentiment Analysis":
            return SentimentAnalysisModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class GuiBase:
    """Base class for GUI components - demonstrates inheritance"""
    
    def __init__(self):
        self._setup_logging()
        self.models = {}
        self._initialize_models()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            self.models["Image Classification"] = ModelFactory.create_model("Image Classification")
            self.models["Sentiment Analysis"] = ModelFactory.create_model("Sentiment Analysis")
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
    
    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """Show message to user"""
        if msg_type == "error":
            messagebox.showerror(title, message)
        elif msg_type == "warning":
            messagebox.showwarning(title, message)
        else:
            messagebox.showinfo(title, message)

class ModelInfoMixin:
    """Mixin class for model information display - demonstrates multiple inheritance"""
    
    def format_model_info(self, model_info: dict) -> str:
        """Format model information for display"""
        info_text = f"Model: {model_info.get('name', 'Unknown')}\n"
        info_text += f"Type: {model_info.get('type', 'Unknown')}\n"
        info_text += f"Category: {model_info.get('category', 'Unknown')}\n"
        info_text += f"Description: {model_info.get('description', 'No description')}\n"
        info_text += f"Input Type: {model_info.get('input_type', 'Unknown')}\n"
        info_text += f"Output Type: {model_info.get('output_type', 'Unknown')}\n"
        info_text += f"Model Size: {model_info.get('model_size', 'Unknown')}\n"
        info_text += f"Use Case: {model_info.get('use_case', 'General')}\n"
        info_text += f"Status: {model_info.get('status', 'Unknown')}\n"
        return info_text

class AIModelGUI(GuiBase, ModelInfoMixin):
    """Main GUI Application - demonstrates multiple inheritance, polymorphism, and encapsulation"""
    
    def __init__(self):
        super().__init__()  # Call parent constructor
        self.root = tk.Tk()
        self.root.title("AI Model Integration GUI")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Private attributes - demonstrates encapsulation
        self._current_model = None
        self._processing = False
        
        self._setup_gui()
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
    
    def _setup_gui(self):
        """Setup the main GUI components"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self._create_model_tab()
        self._create_info_tab()
        self._create_oop_explanation_tab()
    
    def _create_model_tab(self):
        """Create the main model interaction tab"""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="AI Models")
        
        # Title
        title_label = ttk.Label(model_frame, text="AI Model Integration", style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Model selection frame
        selection_frame = ttk.LabelFrame(model_frame, text="Model Selection", padding=10)
        selection_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(selection_frame, text="Select AI Model:").pack(anchor='w')
        self.model_var = tk.StringVar(value="Image Classification")
        model_combo = ttk.Combobox(
            selection_frame, 
            textvariable=self.model_var,
            values=["Image Classification", "Sentiment Analysis"],
            state="readonly"
        )
        model_combo.pack(fill='x', pady=5)
        model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        # Input frame
        input_frame = ttk.LabelFrame(model_frame, text="Input", padding=10)
        input_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Enter image URL or file path:").pack(anchor='w')
        
        # Create a frame for input and browse button
        input_container = ttk.Frame(input_frame)
        input_container.pack(fill='both', expand=True, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(
            input_container, 
            height=8, 
            wrap=tk.WORD,
            font=('Arial', 11)
        )
        self.input_text.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Browse button frame (vertical)
        browse_frame = ttk.Frame(input_container)
        browse_frame.pack(side='right', fill='y')
        
        self.browse_btn = ttk.Button(
            browse_frame,
            text="Browse\nImage",
            command=self._browse_image,
            width=10
        )
        self.browse_btn.pack(pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill='x', pady=5)
        
        self.process_btn = ttk.Button(
            button_frame, 
            text="Process", 
            command=self._process_text
        )
        self.process_btn.pack(side='left', padx=5)
        
        ttk.Button(
            button_frame, 
            text="Clear Input", 
            command=self._clear_input
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame, 
            text="Load Sample Text", 
            command=self._load_sample_text
        ).pack(side='left', padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(model_frame, text="Output", padding=10)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame, 
            height=8, 
            wrap=tk.WORD,
            font=('Arial', 11),
            state='disabled'
        )
        self.output_text.pack(fill='both', expand=True, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            output_frame, 
            mode='indeterminate'
        )
        self.progress.pack(fill='x', pady=5)
    
    def _create_info_tab(self):
        """Create the model information tab"""
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="Model Information")
        
        title_label = ttk.Label(info_frame, text="AI Model Information", style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Model selection for info
        selection_frame = ttk.Frame(info_frame)
        selection_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(selection_frame, text="Select Model for Information:").pack(side='left')
        self.info_model_var = tk.StringVar(value="Image Classification")
        info_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.info_model_var,
            values=["Image Classification", "Sentiment Analysis"],
            state="readonly"
        )
        info_combo.pack(side='left', padx=10)
        info_combo.bind('<<ComboboxSelected>>', self._update_model_info)
        
        # Model info display
        self.model_info_text = scrolledtext.ScrolledText(
            info_frame,
            height=20,
            wrap=tk.WORD,
            font=('Arial', 11),
            state='disabled'
        )
        self.model_info_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Update info initially
        self._update_model_info()
    
    def _create_oop_explanation_tab(self):
        """Create the OOP concepts explanation tab"""
        oop_frame = ttk.Frame(self.notebook)
        self.notebook.add(oop_frame, text="OOP Concepts Explanation")
        
        title_label = ttk.Label(oop_frame, text="Object-Oriented Programming Concepts", style='Title.TLabel')
        title_label.pack(pady=10)
        
        oop_text = scrolledtext.ScrolledText(
            oop_frame,
            wrap=tk.WORD,
            font=('Arial', 11),
            state='normal'
        )
        oop_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add OOP explanation content
        oop_explanation = self._get_oop_explanation()
        oop_text.insert('1.0', oop_explanation)
        oop_text.config(state='disabled')
    
    def _get_oop_explanation(self) -> str:
        """Get OOP concepts explanation text"""
        return """
OOP CONCEPTS IMPLEMENTED IN THIS APPLICATION:

1. ENCAPSULATION:
   - Private attributes in BaseModel class (e.g., self._model_name, self._model)
   - Property decorators for controlled access to private attributes
   - Methods to control access to internal model state

2. INHERITANCE:
   - BaseModel as abstract base class
   - TextGenerationModel and SentimentAnalysisModel inherit from BaseModel
   - GuiBase class provides common functionality for GUI components
   - AIModelGUI inherits from GuiBase

3. MULTIPLE INHERITANCE:
   - ImageClassificationModel and SentimentAnalysisModel inherit from both BaseModel and ModelMixin
   - AIModelGUI inherits from both GuiBase and ModelInfoMixin
   - Demonstrates combining functionality from multiple parent classes

4. POLYMORPHISM:
   - process() method implemented differently in each model class
   - Same method name, different implementations based on model type
   - Factory pattern creates different model instances with same interface

5. METHOD OVERRIDING:
   - get_model_info() method overridden in specific model classes
   - load_model() method customized for each model type
   - Provides specialized behavior while maintaining consistent interface

6. MULTIPLE DECORATORS:
   - @log_method_call decorator for logging method execution
   - @validate_input decorator for input validation
   - Applied to multiple methods demonstrating decorator stacking

7. ABSTRACTION:
   - BaseModel as abstract base class with abstract methods
   - Defines interface that must be implemented by subclasses
   - Hides implementation details while providing consistent interface

8. DESIGN PATTERNS:
   - Factory Pattern: ModelFactory class creates model instances
   - Mixin Pattern: ModelMixin and ModelInfoMixin provide additional functionality
   - Template Method Pattern: Base class defines algorithm structure

IMPLEMENTATION DETAILS:

File Organization:
- base_model.py: Contains base classes, decorators, and mixins
- image_classification_model.py: Implements image classification functionality
- sentiment_analysis_model.py: Implements sentiment analysis functionality
- gui_application.py: Main GUI application with all OOP concepts integrated
- main.py: Entry point that orchestrates the application

Benefits of OOP Approach:
- Code reusability through inheritance
- Maintainable and extensible design
- Clear separation of concerns
- Consistent interface across different model types
- Easy to add new models by extending base classes
"""
    
    def _on_model_change(self, event=None):
        """Handle model selection change - demonstrates event handling"""
        model_name = self.model_var.get()
        if model_name in self.models:
            self._current_model = self.models[model_name]
            self._update_ui_for_model()
    
    def _update_ui_for_model(self):
        """Update UI based on selected model"""
        if self._current_model:
            # Update placeholder text and browse button visibility based on model type
            if isinstance(self._current_model, ImageClassificationModel):
                self.input_text.delete('1.0', tk.END)
                self.input_text.insert('1.0', 'Enter image URL or file path for classification...')
                # Show browse button for image classification
                self.browse_btn.config(state='normal')
            elif isinstance(self._current_model, SentimentAnalysisModel):
                self.input_text.delete('1.0', tk.END)
                self.input_text.insert('1.0', 'Enter text for sentiment analysis...')
                # Hide browse button for sentiment analysis
                self.browse_btn.config(state='disabled')
    
    def _process_text(self):
        """Process text using selected model - demonstrates threading"""
        if self._processing:
            return
        
        input_text = self.input_text.get('1.0', tk.END).strip()
        if not input_text or input_text.startswith('Enter'):
            self.show_message("Input Error", "Please enter some text to process.", "error")
            return
        
        model_name = self.model_var.get()
        if model_name not in self.models:
            self.show_message("Model Error", "Please select a valid model.", "error")
            return
        
        # Start processing in separate thread
        threading.Thread(
            target=self._process_in_thread, 
            args=(input_text, model_name),
            daemon=True
        ).start()
    
    def _process_in_thread(self, input_text: str, model_name: str):
        """Process text in separate thread to avoid GUI freezing"""
        self._processing = True
        self.root.after(0, self._start_processing_ui)
        
        try:
            model = self.models[model_name]
            result = model.process(input_text)  # Polymorphic method call
            self.root.after(0, self._display_result, result)
        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            self.root.after(0, self._display_result, error_msg)
        finally:
            self._processing = False
            self.root.after(0, self._stop_processing_ui)
    
    def _start_processing_ui(self):
        """Update UI when processing starts"""
        self.process_btn.config(text="Processing...", state='disabled')
        self.progress.start()
        
        # Clear output
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', "Processing your request...")
        self.output_text.config(state='disabled')
    
    def _stop_processing_ui(self):
        """Update UI when processing stops"""
        self.process_btn.config(text="Process Text", state='normal')
        self.progress.stop()
    
    def _display_result(self, result: str):
        """Display processing result"""
        self.output_text.config(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', result)
        self.output_text.config(state='disabled')
    
    def _browse_image(self):
        """Open file dialog to browse and select an image file"""
        # Define supported image file types
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.gif *.bmp *.tiff *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("GIF files", "*.gif"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]
        
        try:
            # Open file dialog
            filename = filedialog.askopenfilename(
                title="Select an Image File",
                filetypes=filetypes,
                initialdir=os.path.expanduser("~")  # Start in user's home directory
            )
            
            # If user selected a file, insert its path into the input text
            if filename:
                self.input_text.delete('1.0', tk.END)
                self.input_text.insert('1.0', filename)
                
                # Show confirmation message
                filename_display = os.path.basename(filename)
                self.show_message("File Selected", f"Selected image: {filename_display}")
                
        except Exception as e:
            self.show_message("Browse Error", f"Error browsing for image: {str(e)}", "error")
    
    def _clear_input(self):
        """Clear input text"""
        self.input_text.delete('1.0', tk.END)
    
    def _load_sample_text(self):
        """Load sample text based on selected model"""
        model_name = self.model_var.get()
        if model_name == "Image Classification":
            sample_text = "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=400"
        else:  # Sentiment Analysis
            sample_text = "I really love this product! It works perfectly and exceeded my expectations."
        
        self.input_text.delete('1.0', tk.END)
        self.input_text.insert('1.0', sample_text)
    
    def _update_model_info(self, event=None):
        """Update model information display"""
        model_name = self.info_model_var.get()
        if model_name in self.models:
            model = self.models[model_name]
            model_info = model.get_model_info()
            formatted_info = self.format_model_info(model_info)  # Using mixin method
            
            self.model_info_text.config(state='normal')
            self.model_info_text.delete('1.0', tk.END)
            self.model_info_text.insert('1.0', formatted_info)
            self.model_info_text.config(state='disabled')
    
    def run(self):
        """Start the GUI application"""
        self._on_model_change()  # Initialize with default model
        self.root.mainloop()
