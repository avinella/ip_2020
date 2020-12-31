CXX = g++
CXXFLAGS = -g -Wall -Wextra -pedantic -std=c++1z

all: ipApp

ipApp: main.o ip.o image.o control.o
	$(CXX) -o ipApp $(CXXFLAGS) main.o ip.o image.o control.o

main.o: main.cpp main.h common.h image.h control.h
	$(CXX) -c $(CXXFLAGS) main.cpp

ip.o: ip.cpp ip.h common.h image.h main.h
	$(CXX) -c $(CXXFLAGS) ip.cpp

image.o: image.cpp image.h common.h
	$(CXX) -c $(CXXFLAGS) image.cpp

control.o: control.cpp control.h common.h ip.h main.h
	$(CXX) -c $(CXXFLAGS) image.cpp

clean:
	rm -rf *.o ipApp
