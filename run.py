import uvicorn
import os

if __name__ == "__main__":
    # Ensure fonts directory exists locally
    os.makedirs("fonts", exist_ok=True)
    font_path = os.path.join("fonts", "Nunito-Bold.ttf")
    if not os.path.exists(font_path):
        print("Downloading Nunito-Bold font...")
        import urllib.request
        try:
            urllib.request.urlretrieve(
                "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito-Bold.ttf",
                font_path
            )
            print("Downloaded Nunito-Bold successfully.")
        except Exception as e:
            print(f"Warning: Could not download font: {e}. PIL default font will be used.")

    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}...")
    # Run uvicorn server pointing to app/main.py -> app
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
