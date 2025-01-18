from abc import ABC, abstractmethod


class BaseConvertor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def convert(self, image, output_path):
        pass
