use std::ops::{Range, RangeFrom, RangeTo, RangeFull};

pub struct RangeArg {
    pub start: usize,
    pub end: Option<usize>,
}

impl RangeArg {
    pub fn len(&self, len: usize) -> usize {
        self.end.unwrap_or(len) - self.start
    }
}

impl From<Range<usize>> for RangeArg {
    #[inline]
    fn from(r: Range<usize>) -> RangeArg {
        RangeArg {
            start: r.start,
            end: Some(r.end),
        }
    }
}

impl From<RangeFrom<usize>> for RangeArg {
    #[inline]
    fn from(r: RangeFrom<usize>) -> RangeArg {
        RangeArg {
            start: r.start,
            end: None,
        }
    }
}

impl From<RangeTo<usize>> for RangeArg {
    #[inline]
    fn from(r: RangeTo<usize>) -> RangeArg {
        RangeArg {
            start: 0,
            end: Some(r.end),
        }
    }
}

impl From<RangeFull> for RangeArg {
    #[inline]
    fn from(_: RangeFull) -> RangeArg {
        RangeArg {
            start: 0,
            end: None,
        }
    }
}

impl From<usize> for RangeArg {
    #[inline]
    fn from(i: usize) -> RangeArg {
        RangeArg {
            start: i,
            end: Some(i+1),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[macro_export]
macro_rules! s(
    (@as_expr $e:expr) => ($e);
    (@parse [$($stack:tt)*] $r:expr) => {
        s![@as_expr [$($stack)* s!(@step $r)]]
    };
    (@parse [$($stack:tt)*] $r:expr, $($t:tt)*) => {
        s![@parse [$($stack)* s!(@step $r),] $($t)*]
    };
    (@step $r:expr) => {
        <$crate::RangeArg as ::std::convert::From<_>>::from($r)
    };
    ($($t:tt)*) => {
        s![@parse [] $($t)*]
    };
);

#[test]
fn test_s_macro() {
    let s: [RangeArg; 2] = s![1..3, 1];

    assert!(s[0].start == 1);
    assert!(s[1].start == 1);

    assert!(s[0].end == Some(3));
    assert!(s[1].end == Some(2));

    assert!(s[0].len(5) == 2);
    assert!(s[1].len(5) == 1);
}
