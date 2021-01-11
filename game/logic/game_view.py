import eel
import socket
import imgkit
from datetime import datetime
from pathlib import Path

import numpy as np

from threading import Thread

import uuid

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

""" 
BRON
https://pypi.org/project/Eel/
https://nitratine.net/blog/post/python-gui-using-chrome/
 """


class GameView:
    render: bool = True
    def __init__(self, create_screenshot_check: bool = True, port: int = 80):
        self.create_screenshot_check = create_screenshot_check
        self.reset()
        eel.init(f"{BASE_DIR}/game/web")
        self.start(port)
    
    def reset(self):
        self.iteration: int = 0
        if self.create_screenshot_check:
            self.base_dictionary: str = f"{BASE_DIR}/game_history/{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
            Path(self.base_dictionary).mkdir(parents=True, exist_ok=True)
            Path(f"{self.base_dictionary}/game_gird").mkdir(parents=True, exist_ok=True)
            Path(f"{self.base_dictionary}/queue_shapes").mkdir(parents=True, exist_ok=True)


    def start(self, port):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        eel.start("index.html", port=port, host=ip, block=False)
        eel.sleep(1)


    @eel.expose
    def startBackend(data):
        eel.startFront(data)
    
    def change_game_view(self, grid, score, shape_queue):
        eel.changeGameGrid(grid, score, shape_queue)
    
    @eel.expose
    def change_render_state():
        GameView.render = not GameView.render
        eel.changedRender(GameView.render)


    def create_img(self, type: str, body: str, width: int, uid:str, name: str = "") -> None:
        config = imgkit.config(wkhtmltoimage='C:/Program Files/wkhtmltopdf/bin/wkhtmltoimage.exe')
        options = {'width': width, 'disable-smart-width': ''}
        imgkit.from_string(body, f'{self.base_dictionary}/{type}/{type}_{uid}{name}.jpg', config=config, options=options)

    def create_img_game_grid(self, game_grid, style, uid):
        body: str = style + """<table>"""
        for row in game_grid:
            body += "<tr>"
            for c in row:
                body += f"<td data-shape-nr='{c}'></td>"
            body += "</tr>"
        body += "</table>"
        self.create_img("game_gird", body, 78, uid)
    
    def create_img_shape_queue(self, shape_queue, uid):
        style: str = """<style>
            td{
                border: 1px solid black;
                background-color: black;
            }
            td[data-shape-nr="-1"]{
                background-color: white;
            }
            td[data-shape-nr="0.0"]{
                background-color: white;
            }
        </style>"""
        for index_shape, shape in enumerate(shape_queue):
            standard_shape = np.zeros((5, 5))
            for index_row, row in enumerate(shape.shape):
                for index_col, col in enumerate(row):
                    standard_shape[index_row, index_col] = col
            body = style + "<table>"
            for row in standard_shape:
                body += "<tr>"
                for col in row:
                    body += f"<td data-shape-nr='{col}'></td>"
                body += "</tr>"
            body += "</table>"
            self.create_img("queue_shapes", body, 48, uid, index_shape)

    
    def create_screenshot(self, game_grid, game_queue):
        style: str = """<style>
            td{
                border: 1px solid black;
                background-color: black;
            }
            td[data-shape-nr="-1"]{
                background-color: white;
            }
            td[data-shape-nr="0.0"]{
                background-color: white;
            }
        </style>"""
        uid: str = str(uuid.uuid4())
        t_create_img_game_grid = Thread(target=self.create_img_game_grid, args=(game_grid, style, uid, ))
        t_create_img_shape_queue = Thread(target=self.create_img_shape_queue, args=(game_queue, uid, ))
        t_create_img_game_grid.start()
        t_create_img_shape_queue.start()
        try:
            t_create_img_game_grid.join()
        except:
            pass
        try:
            t_create_img_shape_queue.join()
        except:
            pass
        self.iteration += 1
        return uid


