import tsim

c = tsim.Circuit("H 0")
sampler = c.compile_sampler()

print("tsim.Circuit attributes/methods:")
print(dir(c))
print("\nsampler attributes/methods:")
print(dir(sampler))
