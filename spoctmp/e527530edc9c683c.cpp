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
#include <cctype>
using namespace std;

const int INF = 0x3f3f3f3f;
const long long INFLL = 0x3f3f3f3f3f3f3f3fLL;

inline long long read() {
    long long x = 0, f = 1;
    char ch;
    while (!isdigit(ch)) {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (isdigit(ch)) {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}

const int maxn = 100010;
int m, k;
int d[maxn], mx[maxn], s[maxn];

int main() {
    m = read();
    k = read();
    for (int i = 1; i <= m; ++i) {
        d[i] = read();
    }
    for (int i = 1; i <= m; ++i) {
        s[i] = read();
        mx[i] = max(mx[i - 1], s[i]);
    }
    s[m + 1] = 0; // Handle the s[i+1] for i = m
    int ans = 0, now = s[1];
    for (int i = 1; i <= m; ++i) {
        while (d[i] > now) {
            now += mx[i];
            ans += k;
        }
        now -= d[i];
        now += s[i + 1];
        ans += d[i];
    }
    cout << ans << endl;
    return 0;
}
