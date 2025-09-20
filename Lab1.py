import tensorflow as tf
import numpy as np
import itertools

x = np.array(list(itertools.product([0, 1], repeat=4)), dtype=np.float32)


def logic_func(name, x):
    if name == "AND":
        return np.all(x == 1, axis=1).astype(np.float32)
    elif name == "OR":
        return np.any(x == 1, axis=1).astype(np.float32)
    elif name == "XOR":
        return (np.sum(x, axis=1) % 2).astype(np.float32)
    elif name == "NAND":
        return (1 - np.all(x == 1, axis=1)).astype(np.float32)
    elif name == "NOR":
        return (1 - np.any(x == 1, axis=1)).astype(np.float32)
    elif name == "XNOR":
        return (1 - (np.sum(x, axis=1) % 2)).astype(np.float32)
    else:
        raise ValueError("Невідома функція. Використай: AND, OR, XOR, NAND, NOR, XNOR")

func_name = input("Введіть логічну функцію (AND, OR, XOR, NAND, NOR, XNOR): ").strip().upper()
y = logic_func(func_name, x)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=4, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=300, verbose=0)

loss, acc = model.evaluate(x, y, verbose=0)
print(f"\nФункція: {func_name}")
print("loss:", loss, "accuracy:", acc)

pred = model.predict(x)
for inp, p, expected in zip(x, pred, y):
    print(inp.astype(int), "->", round(p[0]), f"({p[0]:.4f})", "expect", int(expected))
