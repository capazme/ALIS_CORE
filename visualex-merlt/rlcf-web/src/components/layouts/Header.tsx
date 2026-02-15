import { Bell, Search, LogOut, Menu } from 'lucide-react';
import { Button } from '../ui/Button';
import { useUIStore } from '../../app/store/ui';
import { useAuthStore } from '../../app/store/auth';
import { motion } from 'framer-motion';

export function Header() {
  const { notifications, toggleSidebar } = useUIStore();
  const { logout } = useAuthStore();

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <header
      className="flex h-16 items-center justify-between border-b border-slate-800 bg-slate-900/50 px-4 backdrop-blur-sm md:px-6"
      role="banner"
    >
      {/* Mobile menu button + Search Bar */}
      <div className="flex flex-1 items-center gap-3 max-w-md">
        <button
          onClick={toggleSidebar}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 md:hidden"
          aria-label="Toggle navigation menu"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="relative w-full">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search tasks, users, or analytics..."
            aria-label="Search tasks, users, or analytics"
            className="w-full rounded-lg border border-slate-700 bg-slate-800/50 py-2 pl-10 pr-4 text-sm text-slate-200 placeholder-slate-500 transition-colors focus:border-violet-500 focus:outline-none focus:ring-1 focus:ring-violet-500 focus-visible:ring-2 focus-visible:ring-blue-500"
          />
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center space-x-2 md:space-x-3">
        {/* Notifications */}
        <Button
          variant="ghost"
          size="sm"
          className="relative h-9 w-9 p-0 focus-visible:ring-2 focus-visible:ring-blue-500"
          aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ''}`}
        >
          <Bell className="h-4 w-4" />
          {unreadCount > 0 && (
            <motion.span
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs font-medium text-white"
              aria-hidden="true"
            >
              {unreadCount > 9 ? '9+' : unreadCount}
            </motion.span>
          )}
        </Button>

        {/* Logout */}
        <Button
          variant="ghost"
          size="sm"
          onClick={logout}
          className="h-9 w-9 p-0 text-red-400 hover:text-red-300 focus-visible:ring-2 focus-visible:ring-blue-500"
          aria-label="Log out"
        >
          <LogOut className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}
