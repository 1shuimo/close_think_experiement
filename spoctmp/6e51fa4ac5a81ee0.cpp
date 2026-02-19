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
    vector<vector<int>> curr(4, vector<int>(4, 1));
    vector<vector<int>> actions(3, vector<int>(3));
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cin >> actions[i][j];
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (actions[i][j] % 2 == 1) {
                curr[i][j] ^= 1;
                if (j > 0) curr[i][j-1] ^= 1;
                if (i > 0) curr[i-1][j] ^= 1;
                if (j < 2) curr[i][j+1] ^= 1;
                if (i < 2) curr[i+1][j] ^= 1;
            }
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << curr[i][j] << " ";
        }
        cout << endl;
   

```cpp
int main() {
    vector<vector<int>> curr(3, vector<int>(3, 1));
    vector<vector<int>> actions(3, vector<int>(3));
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cin >> actions[i][j];
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (actions[i][j] % 2 == 1) {
                curr[i][j] ^= 1;
                if (j > 0) curr[i][j-1] ^= 1;
                if (i > 0) curr[i-1][j] ^= 1;
                if (j < 2) curr[i][j+1] ^= 1;
                if (i < 2) curr[i+1][j] ^= 1;
            }
        }
    }
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << curr[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}
