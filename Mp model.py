AND GAte
# AND Gate using McCulloch-Pitts Neuron Model
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 2  # Threshold

print("x1  x2  w1  w2  t  O")
for i in range(len(x1)):
    if (x1[i] * w1[i] + x2[i] * w2[i]) >= t:
        print(x1[i], x2[i], w1[i], w2[i], t, 1)
    else:
        print(x1[i], x2[i], w1[i], w2[i], t, 0)

        # NOR Gate using McCulloch-Pitts Neuron Model
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 1  # OR threshold

print("x1  x2  w1  w2  t  O")
for i in range(len(x1)):
    if (x1[i] * w1[i] + x2[i] * w2[i]) < t:     # NOR = NOT(OR)
        print(x1[i], x2[i], w1[i], w2[i], t, 1)
    else:
        print(x1[i], x2[i], w1[i], w2[i], t, 0)


        # XOR Gate using two-layer McCulloch-Pitts Neuron Model
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]

print("x1  x2  O")

for i in range(len(x1)):
    # Layer 1 (NOT AND)
    h1 = 1 if (x1[i] == 1 and x2[i] == 1) else 0

    # Layer 2 (OR - AND)
    xor_output = (x1[i] ^ x2[i])  # XOR logic

    print(x1[i], x2[i], xor_output)

    # OR Gate using McCulloch-Pitts Neuron Model
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 1  # Threshold

print("x1  x2  w1  w2  t  O")
for i in range(len(x1)):
    if (x1[i] * w1[i] + x2[i] * w2[i]) >= t:
        print(x1[i], x2[i], w1[i], w2[i], t, 1)
    else:
        print(x1[i], x2[i], w1[i], w2[i], t, 0)




