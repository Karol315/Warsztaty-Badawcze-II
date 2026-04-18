from abc import ABC, abstractmethod

class BaseAcquisition(ABC):
    @abstractmethod
    def observe(self, true_maze):
        """Aktualizuje wiedzę o środowisku na podstawie obecnej pozycji."""
        pass

    @abstractmethod
    def get_next_move(self, model_uncertainty, R):
        """Zwraca kolejną pozycję (x, y) do zbadania w ramach budżetu R."""
        pass