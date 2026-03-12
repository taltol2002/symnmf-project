import os
import subprocess
import numpy as np

ERROR_MSG = "An Error Has Occurred" 

def get_numpy_expected(data, goal):
    """חישוב מתמטי מדויק דרך Numpy כרפרנס"""
    n = data.shape[0]
    # חישוב מרחקים ו-sym
    dist_sq = np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=2)
    A = np.exp(-dist_sq / 2.0)
    np.fill_diagonal(A, 0)
    
    if goal == "sym":
        return A
    
    # חישוב ddg 
    d = np.sum(A, axis=1)
    if goal == "ddg":
        return np.diag(d)
        
    # חישוב norm 
    if goal == "norm":
        d_inv_sqrt = np.zeros_like(d)
        d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
        D_inv = np.diag(d_inv_sqrt)
        return D_inv @ A @ D_inv
    return None

def run_user_cmd(k, goal, filename):
    """הרצת הקוד של המשתמש בדיוק כמו בדרישות"""
    cmd = ["python3", "symnmf.py", str(k), goal, filename]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res

def parse_output(stdout_str):
    """המרת הפלט המודפס למטריצת Numpy לבדיקה"""
    lines = [line.strip().split(',') for line in stdout_str.strip().split('\n') if line.strip()]
    try:
        return np.array(lines, dtype=float)
    except ValueError:
        return None

def test_accuracy_and_format(data, goal, test_name):
    print(f"Running {test_name} ({goal})...", end=" ")
    filename = "temp_test.txt"
    
    # שומרים את הנתונים ואז טוענים אותם חזרה כדי לעבוד על רמת הדיוק הקטומה 
    np.savetxt(filename, data, delimiter=",", fmt="%.4f") 
    rounded_data = np.loadtxt(filename, delimiter=",")
    
    # התיקון הקריטי: מוודאים שהנתונים תמיד נשארים כמטריצה דו-ממדית, גם אם מדובר בעמודה בודדת
    if rounded_data.ndim == 1:
        rounded_data = rounded_data.reshape(-1, 1)
    
    expected = get_numpy_expected(rounded_data, goal)
    res = run_user_cmd(2, goal, filename)
    
    if res.returncode != 0:
        print(f"❌ FAILED! Program crashed or exited with error code.\nStderr: {res.stderr}")
        return False
        
    actual = parse_output(res.stdout)
    if actual is None:
        print("❌ FAILED! Output format is invalid (not comma separated floats).")
        return False
        
    if actual.shape != expected.shape:
        print(f"❌ FAILED! Shape mismatch. Expected {expected.shape}, Got {actual.shape}.")
        return False
        
    # בדיקת דיוק עד 4 ספרות
    if not np.allclose(actual, expected, atol=1e-4):
        diff = np.abs(actual - expected)
        max_diff = np.max(diff)
        print(f"❌ FAILED! Math discrepancy. Max difference: {max_diff:.6f}")
        return False
        
    print("✅ PASS")
    return True

def test_errors():
    print("Running Error Handling Tests...")
    filename = "temp_test.txt"
    np.savetxt(filename, np.random.rand(5, 3), delimiter=",", fmt="%.4f")
    
    # טסט: k גדול או שווה ל-n
    res = run_user_cmd(5, "symnmf", filename)
    if ERROR_MSG in res.stdout or ERROR_MSG in res.stderr:
        print("  ✅ k >= n : PASS (Error caught)")
    else:
        print("  ❌ k >= n : FAILED (Should print error)")

    # טסט: קובץ לא קיים
    res = run_user_cmd(2, "sym", "not_exists.txt")
    if ERROR_MSG in res.stdout or ERROR_MSG in res.stderr:
        print("  ✅ Bad File : PASS (Error caught)")
    else:
        print("  ❌ Bad File : FAILED (Should print error)")

def main():
    print("=== SYM_NMF EXTREME MASTER TESTER ===")
    print("Building C extension...")
    subprocess.run(["python3", "setup.py", "build_ext", "--inplace"], check=True)
    
    # 1. דאטה סטנדרטי רנדומלי
    standard_data = np.random.rand(10, 5) * 10
    test_accuracy_and_format(standard_data, "sym", "Standard Data")
    test_accuracy_and_format(standard_data, "ddg", "Standard Data")
    test_accuracy_and_format(standard_data, "norm", "Standard Data")
    
    # 2. ממדים גבוהים (לוודא הקצאות זיכרון גדולות)
    high_dim_data = np.random.rand(50, 20) * 5
    test_accuracy_and_format(high_dim_data, "norm", "High Dimensionality (50x20)")
    
    # 3. נתונים במימד אחד (הטסט שקרס קודם)
    one_dim_data = np.random.rand(15, 1) * 10
    test_accuracy_and_format(one_dim_data, "sym", "1D Data (n=15, d=1)")
    
    # 4. נקודות זהות לחלוטין (מרחק אפס)
    identical_data = np.ones((5, 3)) * 4.2
    test_accuracy_and_format(identical_data, "ddg", "Identical Points")
    
    # 5. נקודות מבודדות מאוד שגורמות ל-Underflow (בודק חלוקה באפס ב-C)
    underflow_data = np.array([[0.0, 0.0], [1000.0, 1000.0], [-1000.0, -1000.0]])
    test_accuracy_and_format(underflow_data, "norm", "Isolated Points (Zero Degree Fallback)")
    
    # 6. בדיקת ריצת symnmf 
    print("Running SymNMF Optimization (Checking stability)...", end=" ")
    res = run_user_cmd(2, "symnmf", "temp_test.txt")
    if res.returncode == 0 and ERROR_MSG not in res.stdout and "nan" not in res.stdout.lower():
        print("✅ PASS")
    else:
        print("❌ FAILED! Optimization crashed or printed error.")

    # 7. חסינות לשגיאות
    test_errors()
    
    # ניקוי
    if os.path.exists("temp_test.txt"):
        os.remove("temp_test.txt")
        
    print("=== EXTREME TESTING COMPLETE ===")

if __name__ == "__main__":
    main()