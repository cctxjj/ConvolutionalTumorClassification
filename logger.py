import os

class Logger:

    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(filepath):
            f = open(filepath, 'a')
            f.write("--\n")
            f.close()

    def log(self, message):
        f = open(self.filepath, "a")
        f.write(message)
        f.close()
