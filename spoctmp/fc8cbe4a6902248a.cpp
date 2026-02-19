#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <set>
#include <map>
#include <queue>
#include <stack>
#include <list>
#include <fstream>
#include <climits>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <bitset>
using namespace std;

int main() {
    int curr[4][4] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    int actions[3][3];
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cin >> actions[i][j];
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (actions[i][j] % 2 == 1) {
                curr[i][j] = 1 - curr[i][j];
                
                if (j > 0) {
                    curr[i][j-1] = 1 - curr[i][j-1];
                }
                
                if (i > 0) {
                    curr[i-1][j] = 1 - curr[i-1][j];
                }
                
                if (j < 2) {
                    curr[i][j+1] = 1 - curr[i][j+1];
                }
                
                if (i < 2) {
                    curr[i+1][j] = 1 - curr[i+1][j];
                }
            }
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << curr[i][j];
        }
        std::cout << std::endl;
    }
    
    return 0;
}
