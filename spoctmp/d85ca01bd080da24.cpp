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
    string n1, n2;
    cin >> n1 >> n2;
    for (int i = 1; i < n1.length(); i++) {
        if (n1[i] == n2[i]) {
            n1[i] = '1';
        } else {
            n1[i] = '1';
        }
    }
    cout << n1 << endl;
    return 0;
}
```

```cpp
int main() {
    string n1, n2;
    cin >> n1 >> n2;
    for (int i = 0; i < n1.length(); i++) {
        if (n1[i] == n2[i]) {
            n1[i] = '0';
        } else {
            n1[i] = '1';
        }
    }
    cout << n1 << endl;
    return 0;
}
