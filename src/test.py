import webbrowser
s = ""
a = [1, 4, 6]
b = [2, 7]

for i in a:
    for j in b:
        if (i + j) % 2 == 0:
            s += str(a[(i + 1) % len(a)])
        else:
            s += str(b[(j + 1) % len(b)])

print(s)
url2 = "multisoft.se/" + "466141"
url = "multisoft.se/" +s
webbrowser.open(url)