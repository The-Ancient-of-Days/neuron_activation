from typing import Union, Dict
import numpy as np


class Neuron:
    """Класс, представляющий нейрон с поддержкой различных функций активации."""

    def __init__(
        self, input_size: int, weight_range: tuple = (-0.5, 0.5), seed: int = None
    ):
        """
        Инициализирует нейрон с случайными весами.

        Args:
            input_size: Размер входного вектора.
            weight_range: Диапазон для инициализации весов.
            seed: Seed для воспроизводимости.
        """
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.uniform(weight_range[0], weight_range[1], input_size)

    @staticmethod
    def _all_numbers(array: np.ndarray) -> bool:
        """Проверяет, что все элементы массива — числа."""
        return all(
            isinstance(item, (int, float, np.number)) and not isinstance(item, bool)
            for item in array
        )

    @staticmethod
    def binary(scalar: float, threshold: float = 0) -> int:
        """Бинарная функция активации."""
        return 1 if scalar >= threshold else 0

    @staticmethod
    def sigmoid(scalar: float) -> float:
        """Сигмоидальная функция активации."""
        return 1 / (1 + np.exp(-scalar))

    @staticmethod
    def relu(scalar: float) -> float:
        """Функция активации ReLU."""
        return scalar if scalar > 0 else 0

    @staticmethod
    def leaky_relu(scalar: float, alpha: float = 0.1) -> float:
        """Функция активации LeakyReLU."""
        if alpha <= 0:
            raise ValueError("Alpha must be positive.")
        return scalar if scalar > 0 else scalar * alpha

    @staticmethod
    def elu(scalar: float, alpha: float = 0.1) -> float:
        """Функция активации ELU."""
        if alpha <= 0:
            raise ValueError("Alpha must be positive.")
        return scalar if scalar > 0 else alpha * (np.exp(scalar) - 1)

    def scalar_product(self, features: np.ndarray) -> float:
        """
        Вычисляет скалярное произведение вектора весов и вектора входных данных.

        Args:
            features: Входной вектор признаков.

        Returns:
            Скалярное произведение.

        Raises:
            ValueError: Если размеры массивов не совпадают или элементы не числа.
        """
        if len(features) != len(self.weights) or not self._all_numbers(features):
            raise ValueError(
                "Размеры массивов должны совпадать. Все элементы должны быть числами."
            )
        return np.dot(self.weights, features)

    def activate(
        self,
        scalar: float,
        function: str = "sigmoid",
        all_functions: bool = False,
    ) -> Union[float, Dict[str, float]]:
        """
        Применяет функцию активации к скалярному значению.

        Args:
            scalar: Скалярное значение.
            function: Название функции активации.
            all_functions: Если True, возвращает результаты для всех функций.

        Returns:
            Результат применения функции активации.

        Raises:
            ValueError: Если функция активации неизвестна.
        """
        if all_functions:
            return {
                "linear": scalar,
                "binary": self.binary(scalar),
                "sigmoid": self.sigmoid(scalar),
                "hyperbolic_tangent": np.tanh(scalar),
                "relu": self.relu(scalar),
                "leaky_relu": self.leaky_relu(scalar),
                "elu": self.elu(scalar),
            }
        else:
            function = function.lower().replace(" ", "_")
            if function == "linear":
                return scalar
            elif function == "binary":
                return self.binary(scalar)
            elif function == "sigmoid":
                return self.sigmoid(scalar)
            elif function == "hyperbolic_tangent":
                return np.tanh(scalar)
            elif function == "relu":
                return self.relu(scalar)
            elif function == "leaky_relu":
                return self.leaky_relu(scalar)
            elif function == "elu":
                return self.elu(scalar)
            else:
                raise ValueError(f"Неизвестная функция активации: {function}.")


# Пример использования
if __name__ == "__main__":
    features = np.array([0, 2, 3, 4])
    neuron = Neuron(len(features), seed=42)
    scalar = neuron.scalar_product(features)
    activations = neuron.activate(scalar, all_functions=True)
    print(f"Взвешенная сумма: {scalar:.4f}")
    print("Результаты функций активации:")
    for func, value in activations.items():
        print(f"{func}: {value:.4f}")

