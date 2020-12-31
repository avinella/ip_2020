CXX = g++
CXXFLAGS_C = -g -Wall -Wextra -pedantic -std=c++1z -I"C:\MinGW\freeglut\include"
CXXFLAGS_L = -g -Wall -Wextra -pedantic -std=c++1z -L"C:\MinGW\freeglut\lib" -lfreeglut -lopengl32

all: ipApp

ipApp: main.o ip.o image.o control.o
	$(CXX) -o ipApp $(CXXFLAGS_L) main.o ip.o image.o control.o

main.o: main.cpp main.h common.h image.h control.h
	$(CXX) -c $(CXXFLAGS_C) main.cpp

ip.o: ip.cpp ip.h common.h image.h main.h
	$(CXX) -c $(CXXFLAGS_C) ip.cpp

image.o: image.cpp image.h common.h
	$(CXX) -c $(CXXFLAGS_C) image.cpp

control.o: control.cpp control.h common.h ip.h main.h
	$(CXX) -c $(CXXFLAGS_C) image.cpp

clean:
	rm -rf *.o ipApp
