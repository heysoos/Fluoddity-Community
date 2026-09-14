"""
Debug launcher for Fluoddity - keeps console open on error
"""
import sys
import traceback

def main():
    try:
        # Import and run the main application
        import main
        app = main.App()
        app.run()
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Fluoddity crashed!")
        print("="*60)
        print(f"\nError: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n" + "="*60)
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    # A child process of the frozen build re-runs this exe; without this it
    # would launch a second copy of the app instead of serving its job.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
