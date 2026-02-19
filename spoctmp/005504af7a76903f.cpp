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

long long a, b, c;
long long x[100002];

int main() {
    std::cin >> a >> b;
    for (int i = 1; i < a; ++i) {
        std::cin >> x[i];
        c += (x[i] + b - 1) / b;
    }
    std::cout << (c + 1) / 2 << std::endl;
    return 0;
}
```

```cpp
long long a, b, c;
long long x[100001];

int main() {
    std::cin >> a >> b;
    for (int i = 0; i < a; ++i) {
        std::cin >> x[i];
        c += (x[i] + b - 1) / b;
    }
    std::cout << (c + 1) / 2 << std::endl;
    return 0;
}
