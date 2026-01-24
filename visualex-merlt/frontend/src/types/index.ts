export interface SearchParams {
  act_type: string;
  act_number: string;
  date: string;
  article: string;
  version: 'vigente' | 'originale';
  version_date?: string;
  show_brocardi_info: boolean;
  annex?: string;
  tabLabel?: string;
}
