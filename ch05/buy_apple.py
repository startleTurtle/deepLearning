import layer_naive as multi
apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = multi.MulLayer()
mul_tax_layer = multi.MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)