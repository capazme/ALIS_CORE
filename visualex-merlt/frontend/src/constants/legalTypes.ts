/**
 * Shared legal act type constants for citation correction components.
 * Single source of truth aligned with citationMatcher.ts SUFFIX_TO_ACT_TYPE.
 */

export interface TipoAttoOption {
  value: string;
  label: string;
  full: string;
}

export const TIPO_ATTO_QUICK: TipoAttoOption[] = [
  { value: 'codice civile', label: 'C.C.', full: 'Codice Civile' },
  { value: 'codice penale', label: 'C.P.', full: 'Codice Penale' },
  { value: 'codice di procedura civile', label: 'C.P.C.', full: 'Cod. Proc. Civile' },
  { value: 'codice di procedura penale', label: 'C.P.P.', full: 'Cod. Proc. Penale' },
  { value: 'costituzione', label: 'Cost.', full: 'Costituzione' },
  { value: 'legge', label: 'L.', full: 'Legge' },
  { value: 'decreto legislativo', label: 'D.Lgs.', full: 'Decreto Legislativo' },
  { value: 'decreto legge', label: 'D.L.', full: 'Decreto Legge' },
];

export const TIPO_ATTO_ALL: TipoAttoOption[] = [
  ...TIPO_ATTO_QUICK,

  { value: 'preleggi', label: 'Prel.', full: 'Preleggi (Disp. Prel.)' },
  { value: 'codice della strada', label: 'C.d.S.', full: 'Codice della Strada' },
  { value: 'codice della navigazione', label: 'C.N.', full: 'Codice della Navigazione' },
  { value: 'codice dei contratti pubblici', label: 'C.C.P.', full: 'Codice Contratti Pubblici' },
  { value: 'codice del consumo', label: 'C.d.C.', full: 'Codice del Consumo' },
  { value: "codice dell'amministrazione digitale", label: 'CAD', full: 'Cod. Amm. Digitale' },
  { value: "codice della crisi d'impresa e dell'insolvenza", label: 'CCI', full: 'Cod. Crisi Impresa' },
  { value: 'codice dei beni culturali e del paesaggio', label: 'C.B.C.', full: 'Cod. Beni Culturali' },
  { value: 'codice delle comunicazioni elettroniche', label: 'C.C.E.', full: 'Cod. Comunicazioni Elettr.' },
  { value: 'codice delle assicurazioni private', label: 'C.A.P.', full: 'Cod. Assicurazioni' },
  { value: 'codice antimafia', label: 'C.A.M.', full: 'Codice Antimafia' },
  { value: 'norme in materia ambientale', label: 'T.U.A.', full: 'Testo Unico Ambiente' },
  { value: 'testo unico bancario', label: 'T.U.B.', full: 'Testo Unico Bancario' },
  { value: 'testo unico finanza', label: 'T.U.F.', full: 'Testo Unico Finanza' },
  { value: 'testo unico edilizia', label: 'T.U.E.', full: 'Testo Unico Edilizia' },
  { value: 'codice del terzo settore', label: 'C.T.S.', full: 'Codice Terzo Settore' },
  { value: 'codice del turismo', label: 'C.T.U.', full: 'Codice del Turismo' },
  { value: 'codice proprietà industriale', label: 'C.P.I.', full: 'Cod. Proprietà Industriale' },

  { value: 'disposizioni attuative codice civile', label: 'Disp. Att. C.C.', full: 'Disp. Att. Cod. Civile' },
  { value: 'disposizioni attuative codice procedura civile', label: 'Disp. Att. C.P.C.', full: 'Disp. Att. Cod. Proc. Civ.' },
  { value: 'disposizioni attuative codice penale', label: 'Disp. Att. C.P.', full: 'Disp. Att. Cod. Penale' },
  { value: 'disposizioni attuative codice procedura penale', label: 'Disp. Att. C.P.P.', full: 'Disp. Att. Cod. Proc. Pen.' },

  { value: 'decreto del presidente della repubblica', label: 'D.P.R.', full: 'D.P.R.' },
  { value: 'decreto ministeriale', label: 'D.M.', full: 'Decreto Ministeriale' },
  { value: 'decreto del presidente del consiglio', label: 'D.P.C.M.', full: 'D.P.C.M.' },
  { value: 'regio decreto', label: 'R.D.', full: 'Regio Decreto' },
  { value: 'regio decreto legge', label: 'R.D.L.', full: 'Regio Decreto Legge' },
  { value: 'legge costituzionale', label: 'L.Cost.', full: 'Legge Costituzionale' },
  { value: 'legge regionale', label: 'L.R.', full: 'Legge Regionale' },

  { value: 'regolamento ue', label: 'Reg. UE', full: 'Regolamento UE' },
  { value: 'direttiva ue', label: 'Dir. UE', full: 'Direttiva UE' },
  { value: 'decisione ue', label: 'Dec. UE', full: 'Decisione UE' },

  { value: 'convenzione europea diritti uomo', label: 'CEDU', full: 'Conv. Europea Diritti Uomo' },
  { value: 'trattato sul funzionamento ue', label: 'TFUE', full: 'Trattato Funz. UE' },
];

/** Flat list for select dropdowns (value + label) */
export const TIPO_ATTO_SELECT_OPTIONS = TIPO_ATTO_ALL.map(({ value, full }) => ({
  value,
  label: full,
}));
