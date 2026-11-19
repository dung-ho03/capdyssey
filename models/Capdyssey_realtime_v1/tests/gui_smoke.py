"""Manual/desktop smoke: no hardware and no remote network needed."""
from pathlib import Path
import sys
import tempfile
import tkinter as tk
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop_app import App

with tempfile.TemporaryDirectory() as tmp:
    root = tk.Tk()
    root.withdraw()
    app = App(root)
    app.directory.set(tmp)
    app.start()
    root.after(1200, root.quit)
    root.mainloop()
    root.update()
    assert len(app.session.plot) > 50
    assert len(app.canvas.find_all()) >= 2
    app.stop()
    app.session.reader.join(timeout=2)
    assert not app.session.reader.is_alive()
    root.destroy()
    assert (app.session.directory / 'raw.csv').is_file()
    assert len((app.session.directory / 'raw.csv').read_text().splitlines()) > 50
print('GUI start/draw/stop and demo CSV recording passed')
