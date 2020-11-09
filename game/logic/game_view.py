import eel, socket

""" 
BRON
https://pypi.org/project/Eel/
https://nitratine.net/blog/post/python-gui-using-chrome/
 """

class GameView:
    def __init__(self):
        eel.init("./game/web")
        self.start()
        
    
    def start(self):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        eel.start("index.html", port=80, host=ip, block=False)
    
    def change_game_view(self):
        data = "Test"
        eel.changeGameGrid(data)
        print("send!")
    
    @eel.expose
    def my_python_function(data):
        print(data)



a = GameView()

while True:
    eel.sleep(1)