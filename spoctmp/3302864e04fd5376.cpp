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
    int x1, x2, y1, y2;
    int diff1, diff2;
    cin >> x1 >> y1;
    cin >> x2 >> y2;
    diff1 = abs(x2 - x1);
    diff2 = abs(y2 - y1);
    if (diff1 > diff2) {
        cout << diff1 << endl;
    } else if (diff2 > diff1) {
        cout << diff2 << endl;
    } else {
        cout << diff1 << endl;
    }
    return 0;
}
