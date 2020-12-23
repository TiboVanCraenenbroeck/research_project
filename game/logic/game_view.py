import eel
import socket
import imgkit
from datetime import datetime
from pathlib import Path

import numpy as np
""" 
BRON
https://pypi.org/project/Eel/
https://nitratine.net/blog/post/python-gui-using-chrome/
 """


class GameView:
    def __init__(self):
        self.reset()
        eel.init("./game/web")
        self.start()
    
    def reset(self):
        self.base_dictionary: str = f"./game_history/{datetime.now().strftime('%d_%m_%y__%H_%M%S')}"
        self.iteration: int = 0
        Path(self.base_dictionary).mkdir(parents=True, exist_ok=True)
        Path(f"{self.base_dictionary}/game_gird").mkdir(parents=True, exist_ok=True)
        Path(f"{self.base_dictionary}/queue_shapes").mkdir(parents=True, exist_ok=True)


    def start(self):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        eel.start("index.html", port=80, host=ip, block=False)

    def change_game_view(self, grid, score, shape_queue):
        eel.changeGameGrid(grid, score, shape_queue)
        print("send!")

    @eel.expose
    def startBackend(data):
        eel.startFront(data)
    
    def create_img(self, type: str, body: str, width: int, name: str = "") -> None:
        config = imgkit.config(wkhtmltoimage='C:/Program Files/wkhtmltopdf/bin/wkhtmltoimage.exe')
        options = {'width': width, 'disable-smart-width': ''}
        imgkit.from_string(body, f'{self.base_dictionary}/{type}/{type}_{self.iteration}{name}.jpg', config=config, options=options)

    def create_img_game_grid(self, game_grid, style):
        body: str = style + """<table>"""
        for row in game_grid:
            body += "<tr>"
            for c in row:
                body += f"<td data-shape-nr='{c}'></td>"
            body += "</tr>"
        body += "</table>"
        self.create_img("game_gird", body, 78)
    
    def create_img_shape_queue(self, shape_queue):
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
            self.create_img("queue_shapes", body, 48, index_shape)

    
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
        self.create_img_game_grid(game_grid, style)
        self.create_img_shape_queue(game_queue)
        self.iteration += 1


