import numpy as np
import numba as nb

class Vec2f:
    def __init__(self, x=0, y=0):
        self.array = np.array([x, y], dtype=float)

    def __getitem__(self, index):
        return self.array[index]

    def __sub__(self, other):
        return Vec2f(*self.array - other.array)

    def __add__(self, other):
        return Vec2f(*self.array + other.array)

    def dot(self, other):
        return np.dot(self.array, other.array)

    def length(self):
        return np.linalg.norm(self.array)

    def normalize(self):
        norm = np.linalg.norm(self.array)
        if norm != 0:
            return Vec2f(*self.array / norm)
        return Vec2f()

    def __mul__(self, other):
        if isinstance(other, Vec2f):
            return Vec2f(*self.array * other.array)
        elif isinstance(other, (int, float)):
            return Vec2f(*self.array * other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec2f(*self.array / other)
        else:
            raise TypeError("Unsupported operand type for division.")

class Vec3f:
    def __init__(self, x=0, y=0, z=0):
        self.array = np.array([x, y, z], dtype=float)

    def __getitem__(self, index):
        return self.array[index]

    def __sub__(self, other):
        return Vec3f(*self.array - other.array)

    def __add__(self, other):
        return Vec3f(*self.array + other.array)

    def __neg__(self):
        return Vec3f(-self.array[0], -self.array[1], -self.array[2])

    @staticmethod
    def dot(v1, v2):
        return np.dot(v1.array, v2.array)

    def length(self):
        return np.linalg.norm(self.array)

    def normalize(self):
        norm = np.linalg.norm(self.array)
        if norm != 0:
            return Vec3f(*self.array / norm)
        return Vec3f()

    @staticmethod
    def cross(v1, v2):
        return Vec3f(*np.cross(v1.array, v2.array))

    def __mul__(self, other):
        if isinstance(other, Vec3f):
            return Vec3f(*self.array * other.array)
        elif isinstance(other, (int, float)):
            return Vec3f(*self.array * other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec3f(*self.array / other)
        else:
            raise TypeError("Unsupported operand type for division.")

class Vec3i:
    def __init__(self, x=0, y=0, z=0):
        self.array = np.array([x, y, z], dtype=int)

    def __getitem__(self, index):
        return self.array[index]

    def __sub__(self, other):
        return Vec3i(*self.array - other.array)

    def __add__(self, other):
        return Vec3i(*self.array + other.array)

    def __mul__(self, other):
        if isinstance(other, Vec3i):
            return Vec3i(*self.array * other.array)
        elif isinstance(other, (int, float)):
            return Vec3i(*self.array * other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec3i(*self.array // other)
        else:
            raise TypeError("Unsupported operand type for division.")

class Vec4f:
    def __init__(self, x=0, y=0, z=0, w=0):
        self.array = np.array([x, y, z, w], dtype=float)

    def __getitem__(self, index):
        return self.array[index]

    def __sub__(self, other):
        return Vec4f(*self.array - other.array)

    def __add__(self, other):
        return Vec4f(*self.array + other.array)

    def dot(self, other):
        return np.dot(self.array, other.array)

    def length(self):
        return np.linalg.norm(self.array)

    def normalize(self):
        norm = np.linalg.norm(self.array)
        if norm != 0:
            return Vec4f(*self.array / norm)
        return Vec4f()

    def __mul__(self, other):
        if isinstance(other, Vec4f):
            return Vec4f(*self.array * other.array)
        elif isinstance(other, (int, float)):
            return Vec4f(*self.array * other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec4f(*self.array / other)
        else:
            raise TypeError("Unsupported operand type for division.")



