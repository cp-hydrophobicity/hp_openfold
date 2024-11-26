class LogFile:
    def __init__(self, file_name):
        self.file = None
        self.file = open(file_name, 'w')

    def write(self, text):
        self.file.write(text + '\n')
        self.file.flush()

    def close(self):
        self.file.close()