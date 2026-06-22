from src.calibration import result

xopt = result.x

print("\nOptimal Parameters")
print("-------------------")
print("v0     =", xopt[0])
print("theta  =", xopt[1])
print("rho    =", xopt[2])
print("kappa  =", xopt[3])
print("sigma  =", xopt[4])
