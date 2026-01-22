package Bike 
   package Connectors
    connector EBondCon "Uni-directional bondgraphic connector"
      input Real e "Bondgraphic effort variable";
      output Real f "Bondgraphic flow variable";
      output Real d "Direction variable";
    end EBondCon;

    connector FBondCon "Uni-directional bondgraphic connector"
      output Real e "Bondgraphic effort variable";
      input Real f "Bondgraphic flow variable";
      output Real d "Direction variable";
    end FBondCon;

    connector BondCon "Bi-directional bondgraphic connector"
      output Real e "Bondgraphic effort variable";
      input Real f "Bondgraphic flow variable";
      output Real d "Direction variable";
    end BondCon;

    partial model Port0_3"Partial model invoking three bondgraphic connectors"
      Real e[3] "Bondgraphic effort vector";
      Real f[3] "Bondgraphic flow vector";
      Bike.Connectors.BondCon BondCon1 "First bond graph connector";
      Bike.Connectors.BondCon BondCon2 "Second bond graph connector";
      Bike.Connectors.BondCon BondCon3 "Third bond graph connector";
    equation
      e[1] = BondCon1.e;
      f[1] = BondCon1.d*BondCon1.f;
      e[2] = BondCon2.e;
      f[2] = BondCon2.d*BondCon2.f;
      e[3] = BondCon3.e;
      f[3] = BondCon3.d*BondCon3.f;
    end Port0_3;

    partial model Port1_3 "Partial model invoking three bondgraphic connectors"
      Real e[3] "Bondgraphic effort vector";
      Real f[3] "Bondgraphic flow vector";
      Bike.Connectors.BondCon BondCon1 "First bond graph connector";
      Bike.Connectors.BondCon BondCon2 "Second bond graph connector";
      Bike.Connectors.BondCon BondCon3 "Third bond graph connector";
    equation
      e[1] = BondCon1.d*BondCon1.e;
      f[1] = BondCon1.f;
      e[2] = BondCon2.d*BondCon2.e;
      f[2] = BondCon2.f;
      e[3] = BondCon3.d*BondCon3.e;
      f[3] = BondCon3.f;
    end Port1_3;

    partial model PassivePort "Partial model invoking one bondgraphic connector"
      Real e "Bondgraphic effort";
      Real f "Bondgraphic flow";
      Bike.Connectors.BondCon BondCon1 "Bond graph connector";
    equation
      e = BondCon1.e;
      f = BondCon1.d*BondCon1.f;
    end PassivePort;
  end Connectors;

  package Bonds
    block EBond "One of two causal bond models of the bond graph library"
      Connectors.FBondCon fBondCon1 "Left bond graph connector";
      Connectors.EBondCon eBondCon1 "Right bond graph connector";
    equation
      fBondCon1.e = eBondCon1.e;
      eBondCon1.f = fBondCon1.f;
      fBondCon1.d = -1;
      eBondCon1.d = +1;
    end EBond;

    block FBond"One of two causal bond models of the bond graph library"
      Connectors.EBondCon eBondCon1 "Left bond graph connector";
      Connectors.FBondCon fBondCon1 "Right bond graph connector";
    equation
      fBondCon1.e = eBondCon1.e;
      eBondCon1.f = fBondCon1.f;
      eBondCon1.d = -1;
      fBondCon1.d = +1;
    end FBond;
  end Bonds;

  package Junctions
    model J0_3 "Model of ThreePort 0-junction"
      extends Connectors.Port0_3;
    equation
      e[2:3] = e[1:2];
      sum(f) = 0;
    end J0_3;

    model J1_3 "Model of ThreePort 1-junction"
      extends Connectors.Port1_3;
    equation
      f[2:3] = f[1:2];
      sum(e) = 0;
    end J1_3;
  end Junctions;

  package Elements
  model R "The bondgraphic linear resistor element"
    extends Connectors.PassivePort;
    parameter Real R = 1 "Bondgraphic Resistance";
  equation
    e = R*f;
  end R;
  end Elements;
end Bike;
