def lower_triangle(n):
    print("LOWER TRIANGLE: ")
    for i in range(n):
        for j in range(i+1):
            print("*", end = "")
        print()       
    
def upper_triangle(n):
    print("UPPER TRIANGLE: ")
    for i in range(n):
        for j in range(n+1, i+1, -1):
            print("*", end = "")
        print()
        
def pyramid(p):
    print("PYRAMID: ")
    for i in range((p//2) + 1):
        for j in range((p//2) -i):
            print(" ", end = "")
        for k in range((p//2)-i, (p//2)+i+1):
            print("*", end = "")   
        print()                  


if __name__ == "__main__":
    length = int(input("Enter required length of triangles: "))
    upper_triangle(length)
    lower_triangle(length)
    print(f'''\n# Note: if length is even pyramid is of length+1 
          given length: {length}''')
    pyramid(length)
            