class ThreadsManager:
    def __init__(self, threads) -> None:
        self.threads = threads

    def __enter__(self):
        for t in self.threads:
            t.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.join()
