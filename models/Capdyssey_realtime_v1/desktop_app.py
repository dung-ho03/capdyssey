"""Run on the Windows computer: python desktop_app.py"""
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from realtime.session import Session


class App:
    def __init__(self, root):
        self.root, self.session = root, None
        root.title('Capdyssey 실시간 측정 — 연구용')
        root.geometry('1000x760')
        self.demo = tk.BooleanVar(value=True)
        self.port = tk.StringVar()
        self.url = tk.StringVar()
        self.password = tk.StringVar()
        self.directory = tk.StringVar(value=str(Path.home() / 'Capdyssey_recordings'))
        panel = ttk.Frame(root, padding=15)
        panel.pack(fill='both', expand=True)
        ttk.Label(panel, text='Capdyssey 실시간 PPG', font=('맑은 고딕', 20)).pack(anchor='w')
        ttk.Label(panel, text='예시 모드는 실제 측정·AF 분석을 하지 않습니다. 첫 판정은 10초 수집 후 도착합니다.').pack(anchor='w')
        options = ttk.Frame(panel)
        options.pack(fill='x', pady=8)
        ttk.Checkbutton(options, text='예시 모드', variable=self.demo).pack(side='left')
        ttk.Label(options, text='USB 포트').pack(side='left', padx=8)
        self.ports = ttk.Combobox(options, textvariable=self.port, width=16)
        self.ports.pack(side='left')
        ttk.Button(options, text='포트 새로고침', command=self.refresh).pack(side='left', padx=8)
        for label, var, secret in [('Colab 분석 주소 (미입력 시 수신·저장만)', self.url, False),
                                   ('분석 비밀번호', self.password, True), ('저장 폴더', self.directory, False)]:
            row = ttk.Frame(panel)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label, width=40).pack(side='left')
            ttk.Entry(row, textvariable=var, show='*' if secret else '').pack(side='left', fill='x', expand=True)
        ttk.Button(panel, text='저장 폴더 선택', command=self.choose_folder).pack(anchor='e')
        buttons = ttk.Frame(panel)
        buttons.pack(fill='x', pady=10)
        self.start_button = ttk.Button(buttons, text='측정 시작', command=self.start)
        self.start_button.pack(side='left')
        ttk.Button(buttons, text='중지', command=self.stop).pack(side='left', padx=10)
        self.status = tk.StringVar(value='대기 중')
        self.result = tk.StringVar(value='현재 판정 없음')
        self.details = tk.StringVar(value='')
        ttk.Label(panel, textvariable=self.status).pack(anchor='w')
        self.canvas = tk.Canvas(panel, height=270, background='#101c2a', highlightthickness=0)
        self.canvas.pack(fill='x', pady=8)
        ttk.Label(panel, text='가로: 최근 10초 장치 시간 / 세로: 원본 PPG 값 (자동 축척)').pack(anchor='w')
        ttk.Label(panel, textvariable=self.result, font=('맑은 고딕', 17)).pack(anchor='w', pady=10)
        ttk.Label(panel, textvariable=self.details, wraplength=940).pack(anchor='w')
        self.refresh()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(50, self.tick)

    def refresh(self):
        try:
            from serial.tools import list_ports
            ports = [p.device for p in list_ports.comports()]
            self.ports['values'] = ports
            if ports and not self.port.get(): self.port.set(ports[0])
        except ImportError:
            self.ports['values'] = []

    def choose_folder(self):
        value = filedialog.askdirectory()
        if value: self.directory.set(value)

    def start(self):
        if self.session and not self.session.stop_event.is_set(): return
        if self.session and self.session.reader and self.session.reader.is_alive():
            messagebox.showinfo('종료 중', '이전 장치 연결이 종료될 때까지 잠시 기다려주세요.')
            return
        if not self.demo.get() and not self.port.get().strip():
            messagebox.showerror('USB 포트', '장치를 연결하고 포트를 선택하세요.')
            return
        if not self.demo.get() and self.url.get().strip() and not self.password.get():
            messagebox.showerror('비밀번호', 'Colab에서 생성된 분석 비밀번호를 입력하세요.')
            return
        try:
            self.session = Session(self.directory.get(), demo=self.demo.get(), port=self.port.get().strip(),
                url=self.url.get().strip(), password=self.password.get())
            self.session.start()
        except Exception as e:
            self.session = None
            messagebox.showerror('시작 실패', f'저장 경로와 설정을 확인하세요. ({type(e).__name__})')
            return
        self.start_button.state(['disabled'])
        self.result.set('예시 모드 — 실제 AF 판정 없음' if self.session.demo else '10초 데이터 수집 중')
        self.details.set('저장 위치: ' + str(self.session.directory))
        self.status.set('측정 시작')

    def stop(self):
        if self.session: self.session.stop()
        self.result.set('측정 중지 — 현재 판정 없음')
        self.start_button.state(['!disabled'])

    def close(self):
        self.stop()
        # Let the acquisition context close and flush its CSV before exiting.
        if self.session and self.session.reader and self.session.reader.is_alive():
            self.root.after(100, self.close)
            return
        self.root.destroy()

    def tick(self):
        s = self.session
        if s:
            for _ in range(100):
                try: kind, value = s.events.get_nowait()
                except queue.Empty: break
                if kind == 'result':
                    if s.stop_event.is_set() or value['window_id'] != s.latest_id: continue
                    labels = {'AF':'AF', 'non_AF':'비AF'}
                    text = (labels[value['label']] if value['status'] == 'classified' else
                            '움직임 감지 — 판정 제외' if value['status'] == 'excluded' else '재측정 필요')
                    self.result.set(text)
                    self.details.set(f"분석 구간: 장치 시간 {value['start_s']:.2f}–{value['end_s']:.2f}초 | "
                        f"응답 지연 {value['latency_s']:.1f}초 | 사유 {value.get('reason') or '-'}")
                elif kind == 'window':
                    self.result.set('10초 구간 분석 대기 중' if s.url and not s.demo else '10초 구간 수집 완료')
                elif kind in ('invalid', 'network_error', 'fatal', 'ended', 'demo', 'offline'):
                    self.result.set(value)
                    if kind in ('fatal', 'ended'): self.start_button.state(['!disabled'])
            with s.lock:
                rows = list(s.plot)
                age = time.monotonic() - s.last_sample_wall
            if not s.stop_event.is_set():
                if age > 2:
                    self.result.set('센서 데이터 대기 — 현재 판정 없음')
                self.status.set(('예시 데이터' if s.demo else 'USB 수신') + f' | 화면 샘플 {len(rows)} | 마지막 수신 {age:.1f}초 전')
            self.draw(rows, s.demo)
        self.root.after(50, self.tick)

    def draw(self, rows, demo):
        self.canvas.delete('all')
        if len(rows) < 2: return
        w, h = max(100, self.canvas.winfo_width()), 270
        low, high = min(x.ppg for x in rows), max(x.ppg for x in rows)
        span = max(high-low, 1)
        end = rows[-1].time_s
        points = []
        for r in rows:
            points.extend([w*(r.time_s-(end-10))/10, h-25-(h-50)*(r.ppg-low)/span])
        self.canvas.create_line(*points, fill='#54ded2', width=2)
        self.canvas.create_text(10, 12, anchor='w', fill='white',
            text=f'{"예시 파형" if demo else "PPG RED"} | 최솟값 {low:.0f} / 최댓값 {high:.0f}')


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
