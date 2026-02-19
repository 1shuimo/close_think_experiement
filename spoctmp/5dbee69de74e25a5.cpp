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
    char wb[11][11];
    for (int i = 1; i <= 8; ++i) {
        for (int j = 1; j <= 8; ++j) {
            cin >> wb[i][j];
        }
    }
    int judge = 1;
    for (int i = 1; i <= 8; ++i) {
        for (int j = 1; j <= 7; ++j) {
            if (wb[i][j] == wb[i][j + 1]) {
                judge = 0;
            }
        }
    }
    if (judge == 1) {
        cout << "YES" << endl;
    } else {
        cout << "NO" << endl;
    }
    return 0;
}
```

```cpp
int main() {
    char wb[10][10];
    for (int i = 1; i <= 8; ++i) {
        for (int j = 1; j <= 8; ++j) {
            cin >> wb[i][j];
        }
    }
    int judge = 1;
    for (int i = 1; i <= 8; ++i) {
        for (int j = 1; j <= 7; ++j) {
            if (wb[i][j] == wb[i][j + 1]) {
                judge = 0;
            }
        }
    }
    if (judge == 1) {
        cout << "YES" << endl;
    } else {
        cout << "NO" << endl;
    }
    return 0;
}
