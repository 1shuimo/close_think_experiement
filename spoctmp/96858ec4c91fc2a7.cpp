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

int sum;

int gcd(int a, int b) {
    if (a % b == 0) return b;
    return gcd(b, a % b);
}

void add(int n, int x) {
    while (n != 0) {
        sum += n % x;
        n = n / x;
    }
}

int main() {
    int n;
    cin >> n;
    for (int i = 2; i < n; ++i) {
        add(n, i);
    }
    int div = gcd(sum, n - 2);
    cout << sum / div << '/' << (n - 2) / div << endl;
    return 0;
}
