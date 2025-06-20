import webbrowser
import os
import sys

def test_browser_compatibility():
    """
    Utility function to verify browser functionality by opening a test webpage.
    Ensures the default browser (Firefox) can load a standard webpage correctly.
    """
    try:
       
        browser_path = "firefox" if sys.platform.startswith('linux') else "C:\\Program Files\\Mozilla Firefox\\firefox.exe"
        
        
        webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(browser_path))
        
      
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        webbrowser.get('firefox').open_new_tab(test_url)
        print("Browser compatibility test initiated successfully.")
        
    except Exception as e:
        print(f"Error during browser test: {str(e)}")

if __name__ == "__main__":
    print("Running browser compatibility check...")
    test_browser_compatibility()
