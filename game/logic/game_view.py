import eel
import socket
import imgkit
from datetime import datetime
from pathlib import Path
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
    
    def create_img_test(self, game_grid, game_queue):
        body = """
        <style>
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
        </style>
        <table>"""

        for row in game_grid:
            body += "<tr>"
            for c in row:
                body += f"<td data-shape-nr='{c}'></td>"
            body += "</tr>"
        body += "</table>"
        config = imgkit.config(wkhtmltoimage='C:/Program Files/wkhtmltopdf/bin/wkhtmltoimage.exe')
        options = {'width': 78, 'disable-smart-width': ''}
        imgkit.from_string(body, f'{self.base_dictionary}/game_gird/game_gird_{self.iteration}.jpg', config=config, options=options)
