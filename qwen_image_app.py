"""
Shewbox - standalone desktop chat app for local Ollama models, extended
with local image generation. Works with any installed Ollama model (picked
from Settings); when the active model supports tool calling, it can call
generate_image(), which renders through the local ComfyUI/SDXL backend and
shows the result inline. Persistent chat history lives in chat_history/ and
is listed in the sidebar, same idea as the native Ollama app's history list
but actually wired to the image tool.

Theme tokens are ported 1:1 from the Tu-Torus project's theme.css /
theme-dark.css / themes.css (same --bg/--panel-bg/--text/--accent model).
"""

import json
import os
import shutil
import sys
import threading
import time
import uuid
import tkinter as tk
from tkinter import scrolledtext, font, filedialog, messagebox, simpledialog

from PIL import Image, ImageTk

import ollama
from ollama_image_agent import TOOLS, AVAILABLE_FUNCTIONS, ensure_ollama_running

APP_TITLE = "Shewbox"

if getattr(sys, "frozen", False):
    # Bundled read-only resources (icon) unpack to a temp dir each run.
    RESOURCE_DIR = sys._MEIPASS
    # Writable persistent data must live next to the actual .exe, not in that temp dir.
    DATA_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = RESOURCE_DIR

HISTORY_DIR = os.path.join(DATA_DIR, "chat_history")
ICON_PATH = os.path.join(RESOURCE_DIR, "shewbox.ico")
TEXT_FILE_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".log", ".yaml", ".yml", ".html", ".js", ".ts", ".css"}
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_ATTACHMENT_CHARS = 12000

THEMES = {
    "Light": {
        "bg": "#ffffff", "panel_bg": "#f7f7f8", "border": "#e2e2e5", "control_border": "#8c8c93",
        "text": "#1c1c1f", "text_dim": "#6b6b73", "accent": "#2563eb", "on_accent": "#ffffff",
        "output_bg": "#ffffff", "error": "#b3261e",
    },
    "Dark": {
        "bg": "#1a1a1e", "panel_bg": "#212126", "border": "#33333a", "control_border": "#6d6d75",
        "text": "#e8e8ea", "text_dim": "#97979f", "accent": "#5b8dee", "on_accent": "#0f1a2e",
        "output_bg": "#1a1a1e", "error": "#ff6b64",
    },
    "Forest Green": {
        "bg": "#13201a", "panel_bg": "#182a21", "border": "#2a4033", "control_border": "#5d7867",
        "text": "#dfeee4", "text_dim": "#8fb09b", "accent": "#4caf7a", "on_accent": "#0e1a13",
        "output_bg": "#13201a", "error": "#ff7a6e",
    },
    "Azure Night": {
        "bg": "#0d1b2e", "panel_bg": "#12233a", "border": "#1f3656", "control_border": "#576f8e",
        "text": "#e3ecf7", "text_dim": "#8aa3c2", "accent": "#3d9bff", "on_accent": "#06121f",
        "output_bg": "#0d1b2e", "error": "#ff6b6b",
    },
    "Desert Sunset": {
        "bg": "#fdf6ec", "panel_bg": "#f6e7d3", "border": "#e6cfae", "control_border": "#9a7e61",
        "text": "#3a2a1c", "text_dim": "#735539", "accent": "#ab4a1d", "on_accent": "#ffffff",
        "output_bg": "#fffaf3", "error": "#b3261e",
    },
    "Arctic Dawn": {
        "bg": "#f5f9fc", "panel_bg": "#e9f1f7", "border": "#d3e1ec", "control_border": "#778999",
        "text": "#1b2a36", "text_dim": "#4f6475", "accent": "#16768b", "on_accent": "#ffffff",
        "output_bg": "#fbfdfe", "error": "#b3261e",
    },
    "Plum Midnight": {
        "bg": "#1c1524", "panel_bg": "#231a2e", "border": "#3a2c4a", "control_border": "#746385",
        "text": "#eee6f5", "text_dim": "#a592b8", "accent": "#b77ce8", "on_accent": "#1c1524",
        "output_bg": "#1c1524", "error": "#ff7a8a",
    },
}
DEFAULT_THEME = "Light"
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")


def _load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(data):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _msg_to_plain(m):
    """Best-effort conversion of an ollama Message object (or dict) to a plain dict."""
    if isinstance(m, dict):
        d = dict(m)
    elif hasattr(m, "model_dump"):
        d = m.model_dump(exclude_none=True)
    else:
        d = {"role": getattr(m, "role", "assistant"), "content": getattr(m, "content", "") or ""}
    d.setdefault("content", "")
    if d.get("tool_calls"):
        plain_calls = []
        for c in d["tool_calls"]:
            if isinstance(c, dict):
                plain_calls.append(c)
            elif hasattr(c, "model_dump"):
                plain_calls.append(c.model_dump(exclude_none=True))
        d["tool_calls"] = plain_calls
    return d


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x650")
        if os.path.exists(ICON_PATH):
            try:
                self.root.iconbitmap(ICON_PATH)
            except Exception:
                pass

        os.makedirs(HISTORY_DIR, exist_ok=True)

        self.messages = []
        self._active_turns = {}
        self._discarded_sessions = set()
        self.image_refs = []
        self.pending_attachment = None
        self._save_counter = 0
        self._vision_capable_cache = {}
        self._settings = _load_settings()
        self.theme_name = self._settings.get("theme") if self._settings.get("theme") in THEMES else DEFAULT_THEME
        self.session_id = None
        self.session_title = None
        self.base_font = font.Font(family="Segoe UI", size=11)
        self.settings_win = None

        try:
            ensure_ollama_running()
            self.available_models = [m.model for m in ollama.list().models]
        except Exception:
            self.available_models = ["qwen3:8b"]
        self.model_var = tk.StringVar(value=self.available_models[0] if self.available_models else "qwen3:8b")
        self.theme_var = tk.StringVar(value=self.theme_name)
        self.image_tools_var = tk.BooleanVar(value=self._settings.get("image_tools_enabled", True))

        self._build_layout()
        self.apply_theme(self.theme_name)
        self.refresh_history_list()

    # ---------- layout ----------

    def _build_layout(self):
        self.sidebar = tk.Frame(self.root, width=260)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.main = tk.Frame(self.root)
        self.main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- sidebar header: title + gear ---
        header = tk.Frame(self.sidebar)
        header.pack(fill=tk.X, padx=14, pady=(16, 10))
        self.lbl_title = tk.Label(header, text=APP_TITLE, font=(self.base_font.actual("family"), 15, "bold"))
        self.lbl_title.pack(side=tk.LEFT)
        self.gear_btn = tk.Button(header, text="⚙", command=self.open_settings, relief="flat",
                                   font=(self.base_font.actual("family"), 12), padx=6)
        self.gear_btn.pack(side=tk.RIGHT)

        self.btn_new_chat = tk.Button(self.sidebar, text="+ New Chat", command=self.new_chat,
                                       font=self.base_font, relief="flat", anchor="w", padx=10)
        self.btn_new_chat.pack(fill=tk.X, padx=14, pady=(0, 14))

        self.lbl_history = tk.Label(self.sidebar, text="CHATS", font=(self.base_font.actual("family"), 8, "bold"))
        self.lbl_history.pack(anchor="w", padx=14, pady=(0, 4))

        hist_frame = tk.Frame(self.sidebar)
        hist_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        self.history_list = tk.Listbox(hist_frame, relief="flat", activestyle="none",
                                        font=(self.base_font.actual("family"), 10), highlightthickness=0, borderwidth=0)
        self.history_list.pack(fill=tk.BOTH, expand=True)
        self.history_list.bind("<<ListboxSelect>>", self.on_history_select)
        self.history_list.bind("<Button-3>", self.show_history_context_menu)
        self.history_menu = tk.Menu(self.history_list, tearoff=0)
        self.history_menu.add_command(label="Rename", command=self.rename_history_item)
        self.history_menu.add_command(label="Delete", command=self.delete_history_item)

        self.status_line = tk.Label(self.sidebar, text="", font=(self.base_font.actual("family"), 8),
                                     wraplength=230, justify="left")
        self.status_line.pack(side=tk.BOTTOM, anchor="w", padx=14, pady=14)

        # --- main chat area ---
        self.chat_area = scrolledtext.ScrolledText(
            self.main, wrap=tk.WORD, state="disabled", font=self.base_font, borderwidth=0, padx=14, pady=14,
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))
        self.chat_area.tag_bind("savelink", "<Enter>", lambda e: self.chat_area.config(cursor="hand2"))
        self.chat_area.tag_bind("savelink", "<Leave>", lambda e: self.chat_area.config(cursor=""))

        self.attach_bar = tk.Label(self.main, text="", font=(self.base_font.actual("family"), 9))
        self.attach_bar.pack(fill=tk.X, padx=14)

        input_frame = tk.Frame(self.main)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.attach_btn = tk.Button(input_frame, text="\U0001F4CE", command=self.attach_file,
                                     relief="flat", font=self.base_font, padx=10)
        self.attach_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.input_box = tk.Entry(input_frame, font=self.base_font, relief="flat")
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
        self.input_box.bind("<Return>", lambda e: self.send())

        self.send_btn = tk.Button(input_frame, text="Send", command=self.send, relief="flat", padx=16, font=self.base_font)
        self.send_btn.pack(side=tk.RIGHT)

    def open_settings(self):
        if self.settings_win and tk.Toplevel.winfo_exists(self.settings_win):
            self.settings_win.lift()
            return
        t = THEMES[self.theme_name]
        win = tk.Toplevel(self.root)
        win.transient(self.root)
        win.title("Settings")
        win.geometry("320x260")
        win.configure(bg=t["bg"])
        self.settings_win = win
        if os.path.exists(ICON_PATH):
            try:
                win.iconbitmap(ICON_PATH)
            except Exception:
                pass

        pad = {"padx": 16, "pady": (14, 4)}
        tk.Label(win, text="MODEL", font=(self.base_font.actual("family"), 8, "bold"),
                 bg=t["bg"], fg=t["text_dim"]).pack(anchor="w", **pad)
        model_menu = tk.OptionMenu(win, self.model_var, *self.available_models, command=self.on_model_change)
        model_menu.config(font=self.base_font, bg=t["bg"], fg=t["text"], relief="flat", anchor="w",
                           highlightbackground=t["control_border"])
        model_menu["menu"].configure(bg=t["bg"], fg=t["text"])
        model_menu.pack(fill=tk.X, padx=16)

        tk.Label(win, text="THEME", font=(self.base_font.actual("family"), 8, "bold"),
                 bg=t["bg"], fg=t["text_dim"]).pack(anchor="w", **pad)
        theme_menu = tk.OptionMenu(win, self.theme_var, *THEMES.keys(), command=self.apply_theme)
        theme_menu.config(font=self.base_font, bg=t["bg"], fg=t["text"], relief="flat", anchor="w",
                           highlightbackground=t["control_border"])
        theme_menu["menu"].configure(bg=t["bg"], fg=t["text"])
        theme_menu.pack(fill=tk.X, padx=16)

        image_tools_check = tk.Checkbutton(
            win, text="Enable image tools (generate/edit via ComfyUI)",
            variable=self.image_tools_var, command=self.on_image_tools_toggle,
            bg=t["bg"], fg=t["text"], activebackground=t["bg"], activeforeground=t["text"],
            selectcolor=t["bg"], font=self.base_font, anchor="w",
        )
        image_tools_check.pack(fill=tk.X, padx=16, pady=(14, 4))

    def on_image_tools_toggle(self):
        self._settings["image_tools_enabled"] = self.image_tools_var.get()
        _save_settings(self._settings)

    # ---------- theming ----------

    def apply_theme(self, name):
        t = THEMES[name]
        self.theme_name = name
        self.theme_var.set(name)
        self._settings["theme"] = name
        _save_settings(self._settings)
        self.root.configure(bg=t["bg"])
        self.sidebar.configure(bg=t["panel_bg"])
        self.main.configure(bg=t["bg"])

        for w in (self.lbl_title, self.gear_btn, self.lbl_history):
            w.configure(bg=t["panel_bg"])
        self.lbl_title.configure(fg=t["text"])
        self.gear_btn.configure(fg=t["text"], activebackground=t["accent"])
        self.lbl_history.configure(fg=t["text_dim"])
        self.status_line.configure(bg=t["panel_bg"], fg=t["text_dim"])

        self.btn_new_chat.configure(bg=t["accent"], fg=t["on_accent"], activebackground=t["accent"])

        self.history_list.configure(bg=t["panel_bg"], fg=t["text"], selectbackground=t["accent"],
                                     selectforeground=t["on_accent"], highlightthickness=0)
        self.history_menu.configure(bg=t["panel_bg"], fg=t["text"], activebackground=t["accent"],
                                     activeforeground=t["on_accent"])

        self.chat_area.configure(bg=t["output_bg"], fg=t["text"], insertbackground=t["text"],
                                  selectbackground=t["accent"], selectforeground=t["on_accent"])
        self.chat_area.tag_configure("user", foreground=t["accent"], font=(self.base_font.actual("family"), 11, "bold"))
        self.chat_area.tag_configure("assistant", foreground=t["text"], font=(self.base_font.actual("family"), 11, "bold"))
        self.chat_area.tag_configure("status", foreground=t["text_dim"], font=(self.base_font.actual("family"), 9, "italic"))
        self.chat_area.tag_configure("body", foreground=t["text"])
        self.chat_area.tag_configure("attach", foreground=t["accent"], font=(self.base_font.actual("family"), 9, "italic"))
        self.chat_area.tag_configure("savelink", foreground=t["accent"], underline=True)
        self.chat_area.tag_configure("error", foreground=t["error"])

        self.attach_bar.configure(bg=t["bg"], fg=t["accent"])
        input_bg = t["panel_bg"]
        self.attach_btn.configure(bg=input_bg, fg=t["text"], activebackground=t["accent"])
        self.input_box.configure(bg=input_bg, fg=t["text"], insertbackground=t["text"],
                                  highlightbackground=t["control_border"], highlightcolor=t["accent"], highlightthickness=1)
        self.send_btn.configure(bg=t["accent"], fg=t["on_accent"], activebackground=t["accent"])

        self.update_status()

    def update_status(self):
        self.status_line.configure(text=f"Model: {self.model_var.get()}\nTheme: {self.theme_name}")

    # ---------- persistence / history ----------

    def session_path(self, session_id):
        return os.path.join(HISTORY_DIR, f"{session_id}.json")

    def save_session(self, turn=None):
        session_id = turn["id"] if turn else self.session_id
        messages = turn["messages"] if turn else self.messages
        if not messages or not session_id:
            return
        data = {
            "id": session_id,
            "title": (turn["title"] if turn else self.session_title) or "New chat",
            "model": turn["model"] if turn else self.model_var.get(),
            "updated_at": time.time(),
            "messages": [_msg_to_plain(m) for m in messages],
        }
        with open(self.session_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.refresh_history_list()

    def refresh_history_list(self):
        self.history_list.delete(0, tk.END)
        self._history_ids = []
        files = [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]
        entries = []
        for fn in files:
            try:
                with open(os.path.join(HISTORY_DIR, fn), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                entries.append((data.get("updated_at", 0), data.get("id"), data.get("title", "New chat")))
            except Exception:
                continue
        entries.sort(reverse=True)
        for _, sid, title in entries:
            self.history_list.insert(tk.END, title)
            self._history_ids.append(sid)

    def on_history_select(self, _event):
        sel = self.history_list.curselection()
        if not sel:
            return
        sid = self._history_ids[sel[0]]
        self.load_session(sid)

    def show_history_context_menu(self, event):
        index = self.history_list.nearest(event.y)
        if index < 0 or index >= len(self._history_ids):
            return
        self.history_list.selection_clear(0, tk.END)
        self.history_list.selection_set(index)
        self.history_menu.tk_popup(event.x_root, event.y_root)

    def delete_history_item(self):
        sel = self.history_list.curselection()
        if not sel:
            return
        sid = self._history_ids[sel[0]]
        title = self.history_list.get(sel[0])
        if not messagebox.askyesno("Delete chat", f'Delete "{title}"? This can\'t be undone.'):
            return
        try:
            os.remove(self.session_path(sid))
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))
            return
        if sid == self.session_id:
            self.new_chat(save_current=False)
        if sid in self._active_turns:
            self._discarded_sessions.add(sid)
        self.refresh_history_list()

    def rename_history_item(self):
        sel = self.history_list.curselection()
        if not sel:
            return
        sid = self._history_ids[sel[0]]
        old_title = self.history_list.get(sel[0])
        new_title = simpledialog.askstring("Rename chat", "New name:", initialvalue=old_title, parent=self.root)
        if not new_title or new_title == old_title:
            return
        try:
            with open(self.session_path(sid), "r", encoding="utf-8") as f:
                data = json.load(f)
            data["title"] = new_title
            with open(self.session_path(sid), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showerror("Rename failed", str(e))
            return
        if sid == self.session_id:
            self.session_title = new_title
        if sid in self._active_turns:
            self._active_turns[sid]["title"] = new_title
        self.refresh_history_list()

    def load_session(self, session_id):
        self.save_session()
        try:
            with open(self.session_path(session_id), "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Couldn't open chat", str(e))
            return
        self.session_id = data["id"]
        self.session_title = data.get("title", "New chat")
        self.pending_attachment = None
        self.attach_bar.config(text="")
        active = self._active_turns.get(session_id)
        self.messages = active["messages"] if active else data.get("messages", [])
        if data.get("model") in self.available_models:
            self.model_var.set(data["model"])

        self.render_messages()
        self.send_btn.config(state="disabled" if active else "normal")
        self.update_status()

    def render_messages(self):
        self.chat_area.configure(state="normal")
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.configure(state="disabled")
        self.image_refs = []
        self.replay_messages()

    def refresh_turn_if_current(self, session_id):
        if self.session_id == session_id:
            self.render_messages()

    def finish_turn(self, turn, error=None):
        if turn["id"] not in self._discarded_sessions:
            self.save_session(turn)
        self._discarded_sessions.discard(turn["id"])
        self._active_turns.pop(turn["id"], None)
        if self.session_id == turn["id"]:
            self.messages = turn["messages"]
            self.render_messages()
            if error:
                self.append("error", f"Error: {error}")
            self.send_btn.config(state="normal")

    def replay_messages(self):
        for m in self.messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "user":
                images = m.get("images") or []
                img_path = images[0] if images and os.path.exists(images[0]) else None
                self.append("user", content, img_path)
            elif role == "tool":
                path = content
                if os.path.exists(path):
                    self.append("assistant", "Here you go:", path)
            elif role == "assistant" and content:
                self.append("assistant", content)

    def new_chat(self, save_current=True):
        if save_current:
            self.save_session()
        self.messages = []
        self.session_id = None
        self.session_title = None
        self.pending_attachment = None
        self.attach_bar.config(text="")
        self.chat_area.configure(state="normal")
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.configure(state="disabled")
        self.send_btn.config(state="normal")

    # ---------- behavior ----------

    def on_model_change(self, _value=None):
        self.update_status()
        self.append("status", f"Switched to {self.model_var.get()}")

    def _model_is_vision_capable(self, model):
        if model not in self._vision_capable_cache:
            try:
                info = ollama.show(model)
                self._vision_capable_cache[model] = "vision" in (info.capabilities or [])
            except Exception:
                self._vision_capable_cache[model] = False
        return self._vision_capable_cache[model]

    def attach_file(self):
        path = filedialog.askopenfilename(title="Attach a file")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)
        if ext in TEXT_FILE_EXTENSIONS:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(MAX_ATTACHMENT_CHARS)
            except Exception as e:
                messagebox.showerror("Attach failed", str(e))
                return
            self.pending_attachment = {"kind": "text", "name": name, "content": content}
            self.attach_bar.config(text=f"\U0001F4CE {name} will be sent as context with your next message")
        elif ext in IMAGE_FILE_EXTENSIONS:
            self.pending_attachment = {"kind": "image", "name": name, "path": path}
            self.attach_bar.config(
                text=f"\U0001F5BC {name} will be sent with your next message — ask to edit it or ask about it"
            )
        else:
            messagebox.showinfo(
                "Can't read this file",
                f"{name} isn't a file type this app can read. "
                "Attach a photo (.png, .jpg, .webp, ...) or a text file (.txt, .md, .py, .json, .csv, ...) instead.",
            )

    def append(self, who, text, image_path=None):
        self.chat_area.configure(state="normal")
        if who == "user":
            self.chat_area.insert(tk.END, "You: ", "user")
        elif who == "status":
            self.chat_area.insert(tk.END, text + "\n", "status")
            self.chat_area.configure(state="disabled")
            self.chat_area.see(tk.END)
            return
        elif who == "error":
            self.chat_area.insert(tk.END, text + "\n", "error")
            self.chat_area.configure(state="disabled")
            self.chat_area.see(tk.END)
            return
        else:
            self.chat_area.insert(tk.END, f"{self.model_var.get()}: ", "assistant")
        self.chat_area.insert(tk.END, text + "\n", "body")

        if image_path:
            img = Image.open(image_path)
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)
            self.image_refs.append(photo)
            self.chat_area.image_create(tk.END, image=photo)
            self.chat_area.insert(tk.END, "\n")

            self._save_counter += 1
            tag = f"save_{self._save_counter}"
            self.chat_area.insert(tk.END, "[Save this image...]", ("savelink", tag))
            self.chat_area.insert(tk.END, "\n")
            self.chat_area.tag_bind(tag, "<Button-1>", lambda e, p=image_path: self.save_image(p))

        self.chat_area.insert(tk.END, "\n")
        self.chat_area.configure(state="disabled")
        self.chat_area.see(tk.END)

    def save_image(self, source_path):
        dest = filedialog.asksaveasfilename(
            title="Save image as", defaultextension=".png",
            initialfile=os.path.basename(source_path),
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if dest:
            shutil.copyfile(source_path, dest)
            self.append("status", f"Saved to {dest}")

    def send(self):
        if self.session_id in self._active_turns:
            return
        text = self.input_box.get().strip()
        if not text and not self.pending_attachment:
            return
        self.input_box.delete(0, tk.END)

        if self.session_id is None:
            self.session_id = uuid.uuid4().hex
            self.session_title = text[:48] + ("..." if len(text) > 48 else "")

        outgoing = text
        outgoing_images = None
        if self.pending_attachment:
            attachment = self.pending_attachment
            self.pending_attachment = None
            self.attach_bar.config(text="")

            if attachment["kind"] == "text":
                name, content = attachment["name"], attachment["content"]
                self.append("user", f"\U0001F4CE {name}")
                outgoing = f"[Attached file: {name}]\n{content}\n\n{text}".strip()
            else:
                name, path = attachment["name"], attachment["path"]
                self.append("user", f"\U0001F5BC {name}", image_path=path)
                if self.image_tools_var.get():
                    outgoing = (
                        f"[Attached photo at path: {path}]\n"
                        "(Use this exact path if calling edit_image to modify this photo.)\n\n"
                        f"{text}"
                    ).strip()
                else:
                    outgoing = f"[Attached photo: {name}]\n\n{text}".strip()

                if self._model_is_vision_capable(self.model_var.get()):
                    outgoing_images = [path]

        if text:
            self.append("user", text)

        user_message = {"role": "user", "content": outgoing}
        if outgoing_images:
            user_message["images"] = outgoing_images
        self.messages.append(user_message)
        turn = {"id": self.session_id, "title": self.session_title,
                "model": self.model_var.get(), "messages": self.messages,
                "image_tools": self.image_tools_var.get()}
        self._active_turns[turn["id"]] = turn
        self.send_btn.config(state="disabled")
        threading.Thread(target=self.run_turn, args=(turn,), daemon=True).start()

    def run_turn(self, turn):
        model = turn["model"]
        messages = turn["messages"]
        tools = TOOLS if turn.get("image_tools", True) else None
        error = None
        try:
            ensure_ollama_running()
            try:
                response = ollama.chat(model=model, messages=messages, tools=tools, think=False)
            except Exception:
                response = ollama.chat(model=model, messages=messages, think=False)
            messages.append(response["message"])

            if response["message"].get("tool_calls"):
                self.root.after(0, self.show_status_if_current, turn["id"], "Generating image locally, this can take a minute...")
                ollama.chat(model=model, messages=[], keep_alive=0)

                for call in response["message"]["tool_calls"]:
                    fn_name = call["function"]["name"]
                    fn_args = call["function"]["arguments"]
                    result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)
                    messages.append({"role": "tool", "content": result, "name": fn_name})
                    self.root.after(0, self.refresh_turn_if_current, turn["id"])

                final = ollama.chat(model=model, messages=messages, think=False)
                messages.append(final["message"])
        except Exception as e:
            error = str(e)
        finally:
            self.root.after(0, self.finish_turn, turn, error)

    def show_status_if_current(self, session_id, status):
        if self.session_id == session_id:
            self.append("status", status)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
