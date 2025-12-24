import webview

if __name__ == "__main__":
    webview.create_window("Now Playing", "http://localhost:5432/", fullscreen=True)
    webview.start(gui="gtk")
