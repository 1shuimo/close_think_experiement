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
    string s;
    cin >> s;
    char cur = 'a';
    int mn;
    int ans = 0;
    for (int i = 0; i < s.length(); ++i) {
        mn = min(abs(s[i] - cur), abs(26 - abs(s[i] - cur)));
        ans += mn;
        cur = s[i];
    }
    cout << ans << endl;
    return 0;
}
