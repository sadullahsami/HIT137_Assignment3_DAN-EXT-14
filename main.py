"""
Main entry point for the HIT137 AI Model Integration GUI Application

This file serves as the main program file that brings everything together
and runs the application as required by the assignment.
"""

import sys
import os
import logging
from gui_application import AIModelGUI

def setup_environment():
    """Setup the environment for the application"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'tkinter',
        'transformers',
        'torch'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'tkinter':
                import tkinter
            elif module == 'transformers':
                import transformers
            elif module == 'torch':
                import torch
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("Missing required modules:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nPlease install required dependencies:")
        print("pip install transformers torch")
        print("Note: tkinter is usually included with Python")
        return False
    
    return True

def main():
    """Main function to run the application"""
    print("Starting HIT137 AI Model Integration GUI Application...")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Create and run the GUI application
        app = AIModelGUI()
        
        print("Application initialized successfully!")
        print("Loading GUI interface...")
        print("-" * 60)
        print("OOP Concepts Demonstrated:")
        print("✓ Multiple Inheritance")
        print("✓ Polymorphism")
        print("✓ Encapsulation")
        print("✓ Method Overriding")
        print("✓ Multiple Decorators")
        print("✓ Abstraction")
        print("-" * 60)
        print("AI Models Integrated:")
        print("✓ Image Classification (Vision Transformer)")
        print("✓ Sentiment Analysis (RoBERTa)")
        print("=" * 60)
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        logging.info("Application terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running application: {e}")
        logging.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
    
    print("Application closed successfully.")
    logging.info("Application closed normally")

if __name__ == "__main__":
    main()
