frequency = 1e9  # 1 GHz
operation = 2
area = 63.88 # micro meter square

tops_p_area = (operation * frequency / (area * 1e-6)) / 1e12
print(f"Tops per mm^2: {tops_p_area:.2f} TOPS/mm^2")
