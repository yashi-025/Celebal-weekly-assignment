def lower_triangle(n):
    print("\nLOWER TRIANGLE: ")
    for i in range(n):
        for j in range(i+1):
            print("*", end = "")
        print()       
    
def upper_triangle(n):
    print("\nUPPER TRIANGLE: ")
    for i in range(n):
        for j in range(n, i, -1):
            print("*", end = "")
        print()
        
def pyramid(p):
    print("\nPYRAMID: ")
    for i in range(p):
        #print spaces
        for j in range(p - i):
            print(" ", end = "")
        # print stars
        for k in range(2 * i + 1):
            print("*", end = "")   
        print()                  

if __name__ == "__main__":
    rows = int(input("Enter required no. of rows: "))
    upper_triangle(rows)
    lower_triangle(rows)
    pyramid(rows)
            
