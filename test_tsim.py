if __name__ == "__main__":
    import tsim

    c = tsim.Circuit("H 0\nM 0")
    try:
        sampler = c.compile_sampler()
        samples = sampler.sample(shots=10)
        print("compile_sampler works:")
        print(samples)
    except Exception as e:
        print("compile_sampler failed:", e)

    try:
        c2 = tsim.Circuit("H 0\nM 0\nDETECTOR rec[-1]")
        sampler2 = c2.compile_detector_sampler()
        samples2 = sampler2.sample(shots=10)
        print("compile_detector_sampler works:")
        print(samples2)
    except Exception as e:
        print("compile_detector_sampler failed:", e)
