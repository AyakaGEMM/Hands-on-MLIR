class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0) >;

class Pat<
    dag sourcePattern, dag resultPattern,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)> :
  Pattern<sourcePattern, [resultPattern], additionalConstraints, benefitAdded>;
